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

## Quick start

```bash
pip install torch scikit-learn
python experiment.py                                      # full run, a few min on CPU
python experiment.py --gates 200 --epochs 30 --max-layers 3   # ~20 s smoke test
python experiment.py --skip-e2e                           # greedy + simplification only
python experiment.py --device cuda                        # same experiment on GPU (~10x faster)
python experiment.py --device cuda --max-layers 40 --patience 40 --e2e-depth 40   # depth stress test
```

## Roadmap / open questions

- [x] **Depth stress test**: done — backprop collapses to chance at ~12 layers while greedy keeps learning at 40 (see [above](#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12)). The open half of the question is making that depth *useful*: greedy accuracy still peaks early and decays.
- [ ] **Memory-matched comparison**: give greedy a 4× wider layer (same training memory as end-to-end) and compare accuracy.
- [ ] MNIST / CIFAR-10 on GPU, on top of [difflogic](https://github.com/Felix-Petersen/difflogic) CUDA kernels.
- [ ] Skip connections (concatenate input bits into every layer's wiring pool) to counter information loss from local objectives.
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
