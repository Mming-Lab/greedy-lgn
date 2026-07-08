# greedy-lgn

**Backpropagation-free, layer-by-layer training of Differentiable Logic Gate Networks** — with immediate discretization, adaptive depth, and incremental logic simplification.

> Train one logic layer at a time with a local loss, discretize it, freeze it, and let the next layer learn on *real* 0/1 bits. Stop adding layers when accuracy plateaus. Simplify the circuit as it grows.

Proof-of-concept. Runs on CPU in a few minutes. ~400 lines, no dependencies beyond `torch` and `scikit-learn`.

## Why

[Differentiable Logic Gate Networks (LGNs)](https://github.com/Felix-Petersen/difflogic) learn circuits of 2-input logic gates by relaxing gate choice to a softmax over 16 Boolean functions. They achieve extremely fast, DSP-free inference on FPGAs. But training them end-to-end with backpropagation has three known pain points:

1. **Vanishing gradients in depth.** With the standard parameterization, gradient norms fall below machine precision after ~16 logic layers ([Light DLGN, 2025](https://arxiv.org/abs/2510.03250)). Current fixes (residual initializations, reparameterizations) work *within* the backprop framework.
2. **The discretization gap.** Networks are trained as soft mixtures of gates and discretized afterwards; the mismatch costs accuracy and is an active research topic ([Mind the Gap, 2025](https://arxiv.org/abs/2506.07500)).
3. **Training memory.** Every gate holds 16 float logits, and backprop must keep the whole network soft at once.

This repo explores a different route: **remove backpropagation across layers entirely.**

## Method

```
input bits ──► [train layer 1 (soft, local GroupSum loss)]
                    │ discretize + freeze
                    ▼ hard 0/1 bits
               [train layer 2 on hard bits]
                    │ discretize + freeze
                    ▼
               ... grow until validation accuracy plateaus ...
                    │
                    ▼
               [simplify circuit: constant folding, pass-through
                removal, duplicate merge, dead-gate elimination]
```

- Each layer is trained with **its own local objective** (GroupSum + cross-entropy), in the spirit of greedy layer-wise pretraining, Cascade-Correlation (Fahlman & Lebiere, 1990) and the Forward-Forward algorithm (Hinton, 2022). No gradient ever crosses a layer boundary.
- Because each frozen layer is discretized *before* the next layer trains, later layers learn on genuine Boolean inputs. **The greedy network has zero discretization gap by construction** — the reported accuracy *is* the accuracy of the final hard circuit.
- Only **one layer is ever soft** during training: float memory for gate logits is `gates × 16` instead of `gates × 16 × depth`.
- Depth is **not a hyperparameter**: layers are added until the hard-probe validation accuracy stops improving.
- After training, a simplification pass (constant folding → pass-through/NOT reduction → duplicate merge → dead-gate elimination) shrinks the circuit and is **verified to be bit-exact** against the original.

## Honest results (toy scale)

`sklearn` digits (8×8, thermometer-binarized to 192 bits), 500 gates/layer, CPU:

| | greedy (this repo) | end-to-end backprop |
|---|---|---|
| depth | **4 (chosen automatically)** | 4 (copied from greedy) |
| hard-circuit test accuracy | 88.2% | **93.6%** |
| discretization gap | **0 (by construction)** | 0.0 at this scale¹ |
| float logits held during training | **8,000 (one layer)** | 32,000 (×4) |
| circuit after simplification | 2,000 → **1,316 gates (65.8%)**, bit-identical | — |

¹ With a smaller/undertrained config (`--gates 200 --epochs 30`), the end-to-end baseline shows a **+8.2 pt discretization gap** while greedy remains at exactly 0. At convergence on this easy dataset the gap closes; literature reports it re-appearing at larger depth/scale.

Full run log: see [issue #1](https://github.com/Mming-Lab/greedy-lgn/issues/1)

**The takeaway is mixed, and that's the point.** Local training currently loses ~5 pt of accuracy to backprop — consistent with the Forward-Forward literature. In exchange you get zero discretization gap, ~1/depth training memory, automatic depth, and a circuit you can simplify incrementally. Whether that trade is worth it is exactly what the roadmap below is for.

Also observed: **duplicate-gate merging found 0 duplicates** — with fixed random wiring, two gates almost never share both inputs. The real simplification wins are pass-through and dead-gate removal (34% of gates here). If you came for De Morgan-style rewriting, this is the empirical answer.

## Depth stress test: greedy survives 40 layers, backprop dies at ~12

The first roadmap item, answered. We force greedy training to grow 40 layers (`--max-layers 40 --patience 40`) and train end-to-end baselines at fixed depths (`--e2e-depth N`). Same 500 gates/layer, RTX 3060 Laptop GPU. Hard-circuit test accuracy:

| depth | greedy (hard probe at that layer) | end-to-end backprop (discretized) |
|---|---|---|
| 4 | **88.2%** (peak) | 93.6% |
| 8 | 84.9% | 90.9% (a +1.3 pt discretization gap appears) |
| 12 | 82.4% | **10.7% — chance level** |
| 16 | 76.9% | 10.2% |
| 24 | 71.3% | 10.0% |
| 32 | 64.2% | 10.4% |
| 40 | 56.0% | 10.0% |

Three observations, stated honestly:

1. **End-to-end backprop collapses to chance between depth 8 and 12** and never recovers. This is vanishing gradients, not undertraining: quadrupling the training budget at depth 40 (1,200 epochs) still gives 10.2%. Consistent with [Light DLGN](https://arxiv.org/abs/2510.03250)'s report of gradient norms below machine precision by ~16 layers — our layers are narrower (500 gates), which plausibly moves the cliff earlier.
2. **Greedy training never stops learning.** At layer 40 the local objective still reaches 69.9% train / 56.0% test. No gradient ever crosses a layer boundary, so there is no depth at which the training signal can die.
3. **Caveat: surviving depth ≠ exploiting depth.** Greedy's accuracy peaks at depth 4 and decays monotonically afterwards — each additional hard layer loses information (no skip connections yet). This experiment supports "greedy can train at any depth", not "deeper greedy networks are better". Turning survivable depth into *useful* depth is the skip-connections item in the roadmap below.

Full run log: see [issue #1](https://github.com/Mming-Lab/greedy-lgn/issues/1).

## Memory-matched comparison: equal training memory, greedy wins

Greedy's training-memory advantage (only one layer is ever soft) can be spent on width instead. With 4× wider layers (2,000 gates), greedy holds the same 32,000 float logits during training as the 4-layer end-to-end baseline:

| config | float logits during training | hard-circuit test acc (seeds 1/2/3) | mean |
|---|---|---|---|
| greedy, 500 gates/layer | 8,000 | 88.2 / 88.0 / 88.9 | 88.4% |
| greedy, 1,000 gates/layer | 16,000 | 92.7 (seed 1 only) | — |
| **greedy, 2,000 gates/layer** | **32,000** | **94.7 / 95.3 / 94.9** | **95.0%** |
| end-to-end, 500 × 4 layers | 32,000 | 93.6 / 90.4 / 90.4 | 91.5% |

- **At equal training memory, greedy beats end-to-end on every seed tested** (mean +3.5 pt) and with much lower variance (0.6 pt spread vs 3.2 pt). Depth is still chosen automatically (4 on all seeds) and the discretization gap is still structurally zero, while e2e shows small seed-dependent gaps (e.g. +0.9 pt on seed 2).
- **The honest cost: a larger inference circuit.** The memory-matched greedy circuit is ~5,300 gates after simplification vs 2,000 (raw) for e2e — greedy trades hardware area for training memory and cross-seed stability. (The simplification pass currently runs only in the greedy pipeline, so the e2e count is unsimplified.)
- Same toy-scale caveats as above: one easy dataset, 450 test samples, 3 seeds.

Full run logs: see [issue #1](https://github.com/Mming-Lab/greedy-lgn/issues/1).

## Skip connections: re-exposing the input turns survivable depth into usable depth

Classic residual addition (`x + f(x)`) does not exist in Boolean circuits, but its cheapest circuit-native analogue does: with `--skip-input`, every layer's random wiring pool becomes `[input bits ∥ previous layer]` instead of the previous layer alone. Zero extra gates — it is only wiring. This directly attacks the information loss that made greedy accuracy decay with depth:

| depth | greedy, no skip | greedy, `--skip-input` |
|---|---|---|
| 4 | **88.2%** (peak) | 87.1% |
| 8 | 84.9% | **90.4% (peak)** |
| 12 | 82.4% | 90.0% |
| 20 | 74.0% | 88.4% |
| 30 | 61.8% | 86.2% |
| 40 | 56.0% | 83.6% |

(500 gates/layer, growth forced to 40 layers, seed 1.)

- **The depth decay is largely gone** (layer 40: 56.0% → 83.6%), and for the first time **depth actually helps**: the peak moves from 88.2% at depth 4 to 90.4% at depth 8 (+2.2 pt). The depth-stress-test caveat above — "surviving depth ≠ exploiting depth" — is now half-answered.
- **Combined with the memory-matched width** (2,000 gates/layer): **95.6 / 95.6 / 96.0% over seeds 1/2/3, mean 95.7%** — the best result in this repo, vs 91.5% mean for end-to-end at equal training memory. Depth is still chosen automatically (7/5/3), gap still structurally zero, simplification still verified bit-exact.
- **A hypothesis that did *not* survive the data**: we expected skip wiring to free the ~20% of gates that simplification reveals as pass-throughs (gates that only copy bits forward). The pass-through fraction stayed at ~20% with skip enabled. The benefit is information access, not gate savings — though skip circuits do simplify harder overall (47.8% of gates kept vs 65.8% without skip at the respective peaks).
- The e2e baseline is unchanged by this flag (standard DLGN wiring); same toy-scale caveats as above.

**DenseNet-style variant (`--skip-all`, negative result reported for honesty):** exposing *all* previous layers (not just the input) gives the flattest depth curve of all — 88.4% at layer 40, best 89.8% at depth 29 — but never beats `--skip-input`'s peak (90.4%), and at memory-matched width it is slightly *worse* (95.1% vs 95.7% mean over 3 seeds), plausibly because the ever-growing pool dilutes the random wiring. One striking side effect: dense circuits simplify dramatically harder — the 40-layer network shrinks to **23.8%** of its gates (14,500 → 3,457, mostly dead-gate elimination), since later layers cherry-pick the useful bits of the whole history. `--skip-input` remains the recommended configuration.

Full run logs: see [issue #1](https://github.com/Mming-Lab/greedy-lgn/issues/1).

## Quick start

```bash
pip install torch scikit-learn
python experiment.py                                      # full run, a few min on CPU
python experiment.py --gates 200 --epochs 30 --max-layers 3   # ~20 s smoke test
python experiment.py --skip-e2e                           # greedy + simplification only
python experiment.py --device cuda                        # same experiment on GPU (~10x faster)
python experiment.py --device cuda --max-layers 40 --patience 40 --e2e-depth 40   # depth stress test
python experiment.py --device cuda --gates 2000 --skip-input --max-layers 16 --skip-e2e   # best config (95.7% mean)
```

## Roadmap / open questions

- [x] **Depth stress test**: done — backprop collapses to chance at ~12 layers while greedy keeps learning at 40 (see [above](#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12)). The open half of the question is making that depth *useful*: greedy accuracy still peaks early and decays.
- [x] **Memory-matched comparison**: done — at equal training memory (4× wider layers), greedy outperforms end-to-end on all seeds tested, 95.0% vs 91.5% mean (see [above](#memory-matched-comparison-equal-training-memory-greedy-wins)).
- [ ] MNIST / CIFAR-10 on GPU, on top of [difflogic](https://github.com/Felix-Petersen/difflogic) CUDA kernels.
- [x] Skip connections: done — `--skip-input` removes most of the depth decay (layer 40: 56.0% → 83.6%) and moves the accuracy peak deeper (see [above](#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth)). The DenseNet-style refinement (`--skip-all`) was also tested: flattest depth curve, but no accuracy win — `--skip-input` stays the default recommendation.
- [ ] Better local objectives: Forward-Forward goodness on binary vectors, [Mono-Forward](https://arxiv.org/abs/2501.09238)-style projection losses.
- [ ] Simplify *between* growth steps (currently done once at the end) and rewire the next layer to the simplified circuit.
- [ ] Export simplified circuits to Verilog / run through ABC for comparison with proper logic synthesis.

## Related work

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) (Petersen et al., NeurIPS 2022) and [difflogic](https://github.com/Felix-Petersen/difflogic)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732) (NeurIPS 2024) — includes post-training logic synthesis
- [Light Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250) (2025) — depth via reparameterization (the backprop-side answer to the same problem)
- [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) (Hinton, 2022)
- Cascade-Correlation (Fahlman & Lebiere, 1990) — the original "grow and freeze" network
- To our knowledge, **the combination** (LGN × backprop-free layer-wise training × adaptive depth × incremental simplification) has no published precedent as of mid-2026. If you know of one, please open an issue.

## License

MIT

---

## 日本語概要

論理ゲートネットワーク(DLGN)を**逆伝播なしで1層ずつ**学習する実証実験です。各層をローカルな損失(GroupSum+交差エントロピー)で学習したら**即座に離散化して凍結**し、次の層は本物の0/1ビットの上で学習します。検証精度が頭打ちになったら層の追加を止めるため、深さはハイパーパラメータではなく自動決定されます。学習後(将来的には成長の途中)に定数畳み込み・パススルー除去・重複マージ・デッドゲート削除で回路を簡略化し、元の回路と完全に同一の出力を返すことを検証します。

現状の結果は正直に言って一長一短です:精度はend-to-end逆伝播に約5ポイント負けますが、離散化ギャップが構造的にゼロ、学習メモリが深さ分の1、深さの自動決定、という利点があります。この交換が割に合う条件(深い回路・メモリ制約下)を探すのが上記ロードマップです。CPUで数分で再現できます。

深さストレステスト(RTX 3060)では、**逆伝播は12層でチャンスレベル(10%)に崩壊**し、エポックを4倍にしても回復しませんでした(勾配消失)。一方**greedyは40層目でも学習が成立**します(train 69.9%)。ただしテスト精度のピークは深さ4のままで、深さを積むほど情報損失により単調劣化するため、「深さで生き残れる」と「深さを活かせる」は別問題です。後者の解決(skip connections)が次の課題です。

メモリ等価比較では、greedyの層幅を4倍(2,000ゲート)にして学習時のfloatロジット数をe2eと同じ32,000個に揃えたところ、**テスト精度は3シード全てでe2eを上回りました**(平均95.0% vs 91.5%、バラつきもgreedyの方が小さい)。正直なコストとして、推論回路は簡略化後でも約5,300ゲートとe2e(2,000ゲート)の約2.7倍になります — 学習メモリと安定性をハードウェア面積で買うトレードオフです。

skip connections(`--skip-input`: 各層の配線候補に元の入力192ビットを常に含める。ゲート数は増えず配線のみ)では、深さによる劣化がほぼ解消しました(40層目: 56.0%→83.6%)。ピークも88.2%(深さ4)から90.4%(深さ8)に上がり、**初めて「深さが精度に貢献する」結果**が出ています。幅4倍と組み合わせると3シード平均**95.7%**で、本リポジトリの現時点の最良値です。
