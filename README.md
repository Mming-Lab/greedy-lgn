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

- Each layer is trained with **its own local objective** (GroupSum + cross-entropy), in the spirit of greedy layer-wise pretraining, Cascade-Correlation (Fahlman & Lebiere, 1990) and the Forward-Forward algorithm (Hinton, 2022). No gradient ever crosses a frozen layer boundary.
- Because each frozen layer is discretized *before* the next layer trains, later layers learn on genuine Boolean inputs. **The greedy network has zero discretization gap by construction** — the reported accuracy *is* the accuracy of the final hard circuit.
- Only **one training window is ever soft**: float memory for gate logits is `gates × 16 × window` instead of `gates × 16 × depth` (default window = 1 layer).
- Depth is **not a hyperparameter**: layers are added until the hard-probe validation accuracy stops improving.
- After training, a simplification pass (constant folding → pass-through/NOT reduction → duplicate merge → dead-gate elimination) shrinks the circuit and is **verified to be bit-exact** against the original.
- Optional extensions, all off by default: `--skip-input` (re-expose input bits to every layer's wiring pool), `--window W --commit J` (train W layers ahead with backprop bounded to the window, freeze J at a time).

## Headline result

`sklearn` digits (8×8, thermometer-binarized to 192 bits), 500 gates/layer, CPU:

| | greedy (this repo) | end-to-end backprop |
|---|---|---|
| depth | **4 (chosen automatically)** | 4 (copied from greedy) |
| hard-circuit test accuracy | 88.2% | **93.6%** |
| discretization gap | **0 (by construction)** | 0.0 at this scale |
| float logits held during training | **8,000 (one layer)** | 32,000 (×4) |
| circuit after simplification | 2,000 → **1,316 gates (65.8%)**, bit-identical | — |

**The takeaway is mixed, and that's the point.** Plain local training loses ~5 pt of accuracy to backprop — in exchange for zero discretization gap, ~1/depth training memory, automatic depth, and a simplifiable circuit. The experiments below chip away at that 5 pt from different angles; most of it turned out to be recoverable.

## All experiments (details in [RESULTS.md](RESULTS.md))

| experiment | headline | details |
|---|---|---|
| Depth stress test | e2e collapses to chance at ~12 layers (vanishing gradients); greedy still learns at layer 40 | [→](RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12) |
| Memory-matched width | at equal training memory (4× wider layers), greedy **beats** e2e: 95.0% vs 91.5% mean, 3 seeds | [→](RESULTS.md#memory-matched-comparison-equal-training-memory-greedy-wins) |
| Skip connections (`--skip-input`) | depth finally pays: peak 88.2%@4 → 90.4%@8; with 4× width 95.7% mean. DenseNet-style `--skip-all` tested, negative | [→](RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth) |
| MNIST first pass | the pattern replicates at 45× the data: memory-matched greedy+skip 84.6% vs e2e 80.1% (absolute numbers far below difflogic-scale budgets, stated honestly) | [→](RESULTS.md#mnist-the-pattern-replicates-first-pass-small-budget) |
| Windowed lookahead (`--window`) | training 2 layers ahead closes ~⅔ of the myopia gap: 90.4% vs e2e's 91.5% (3 seeds), +2.4 pt on MNIST; overlap/receding-horizon variant loses to plain blocks | [→](RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap) |
| Ensemble voting (`--ensemble`) | parallel hard circuits + vote: stacks with everything (digits **96.4% — repo best**); on MNIST 4×500-gate members beat the single 2,000-gate best (84.7% vs 84.6%) at **half the training memory**; not a substitute for direct width | [→](RESULTS.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever) |
| MNIST scaling | width is the dominant lever (4,000 gates: 89.8% single) and ensembling stacks: **90.9%** with 4×4,000+skip — first crossing of 90%; more epochs and window×width confirmed dead (+0.1 pt each); 8,000 gates OOMs at 6 GB (pools, not eval temporaries) | [→](RESULTS.md#mnist-scaling-width--ensembles-push-past-90) |
| Forward-Forward objective (`--objective ff`) | goodness = **popcount** on binary layers, so the whole FF inference is one logic circuit; 2.4 pt behind supervised local CE on digits (86.0%) but **+2.5 pt ahead on MNIST** (76.8%) — and the first lever to exploit depth (17 layers) without skip wiring; needs label-bit replication for sparse random wiring. Windowed lookahead **stacks with FF** (digits 88.0%, MNIST 78.2% — best 500-gate single net) unlike with skip | [→](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |

Full run logs (environment, commands, raw output): **one GitHub issue per experiment** ([#1](https://github.com/Mming-Lab/greedy-lgn/issues/1) main run … [#7](https://github.com/Mming-Lab/greedy-lgn/issues/7) windowed lookahead), linked from each [RESULTS.md](RESULTS.md) section.

## Quick start

```bash
pip install torch scikit-learn
python experiment.py                                      # full run, a few min on CPU
python experiment.py --gates 200 --epochs 30 --max-layers 3   # ~20 s smoke test
python experiment.py --skip-e2e                           # greedy + simplification only
python experiment.py --device cuda                        # same experiment on GPU (~10x faster)
python experiment.py --window 2 --commit 2 --win-loss all # 2-layer lookahead blocks (+2 pt)
python experiment.py --ensemble 4 --skip-e2e              # 4 independent nets + voting
python experiment.py --objective ff --ff-label-rep 38 --skip-e2e   # Forward-Forward objective
python experiment.py --device cuda --gates 2000 --skip-input --max-layers 16 --skip-e2e --ensemble 4   # best config (96.4%)
python experiment.py --dataset mnist --device cuda --batch 4096 --epochs 30 --gates 2000 --skip-input --max-layers 10 --skip-e2e   # MNIST (GPU recommended)
```

## Roadmap / open questions

- [x] **Depth stress test** — backprop dies at ~12 layers, greedy survives 40 ([details](RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12)).
- [x] **Memory-matched comparison** — greedy wins at equal training memory ([details](RESULTS.md#memory-matched-comparison-equal-training-memory-greedy-wins)).
- [x] **Skip connections** — `--skip-input` makes depth useful; `--skip-all` negative ([details](RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth)).
- [x] **MNIST first pass** — pattern replicates; absolute accuracy still small-budget ([details](RESULTS.md#mnist-the-pattern-replicates-first-pass-small-budget)).
- [x] **Windowed lookahead** — `--window 2` recovers most of the myopia deficit; window > 2 and overlapping commits don't help ([details](RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap)).
- [x] **Ensemble voting** — `--ensemble M` stacks with every other lever; repo best on digits (96.4%) and the training-memory-free path to MNIST scaling ([details](RESULTS.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever)).
- [x] **MNIST scaling, first round** — width × ensembles reaches 90.9%; epochs and window×width are dead ends ([details](RESULTS.md#mnist-scaling-width--ensembles-push-past-90)).
- [ ] MNIST absolute accuracy, next levers: 8,000-gate layers, better input binarization, convolutional wiring.
- [x] **Forward-Forward objective** — popcount goodness works: behind supervised local CE on digits, ahead on MNIST, and exploits depth without skip wiring ([details](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist)).
- [ ] [Mono-Forward](https://arxiv.org/abs/2501.09238)-style projection losses; FF × width/ensemble combinations (FF × window already tested: it stacks).
- [ ] CIFAR-10 / larger widths, on top of [difflogic](https://github.com/Felix-Petersen/difflogic) CUDA kernels.
- [ ] Simplify *between* growth steps (currently done once at the end) and rewire the next layer to the simplified circuit.
- [ ] Export simplified circuits to Verilog / run through ABC for comparison with proper logic synthesis.

## What is new here — and what is not

Being explicit about the boundary, because most ingredients of this repo are borrowed. The **single novel primitive** is:

> *Train one logic layer with a local loss, discretize it immediately, freeze it, and train the next layer on the real 0/1 bits.*

Everything this repo claims either derives from that primitive or is inherited from prior work:

| property | status |
|---|---|
| No multipliers / DSPs / floats; maps to FPGA LUTs | **Inherited** from LGNs ([difflogic](https://github.com/Felix-Petersen/difflogic)) — not our contribution, just the platform. |
| **Zero discretization gap** | **Direct consequence of the primitive.** Within the LGN literature we find no precedent (as of mid-2026) for a training procedure in which the gap is structurally zero rather than minimized after the fact. Precise claim: not "the first verified-equals-deployed network" (exact-by-construction routes exist outside LGNs, e.g. LogicNets' truth-table enumeration), but *an LGN that stays bit-exact throughout training*. |
| Training memory = one layer, not depth | **Not unique** — any greedy layer-wise scheme (Cascade-Correlation, Forward-Forward) has this. What *is* unique is its intersection with bit-exactness: frozen layers could be burned to an FPGA and the next layer trained on the physical chip's outputs (**hardware-in-the-loop growth**). Backprop cannot do this (gradients don't cross silicon); continuous-activation local methods can't either (chip outputs ≠ training-time activations). Currently a possibility, not a demonstrated result. |
| Adaptive depth / grow-and-freeze | **Cascade-Correlation heritage (1990)** — not new as an idea. Grounding it in circuits adds one genuinely new reading: since circuit depth = critical-path latency, stopping at the accuracy plateau automatically yields a *minimum-latency* circuit for that accuracy. Post-deployment growth (adding layers onto a frozen deployed circuit) is likewise possible in principle — but our own depth-stress data shows added depth only pays off with skip wiring, so we treat it as speculative. |
| Windowed lookahead (`--window`) | **Block-wise greedy training exists** (Belilovsky et al., 2019, with auxiliary heads). What is added here: the blocks are discretized and frozen as they are committed (bit-exact prefix preserved), depth stays adaptive, and the overlap ablation (commit < window, receding-horizon style) is reported — including the negative result that overlap does not beat plain blocks. |

## Related work

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) (Petersen et al., NeurIPS 2022) and [difflogic](https://github.com/Felix-Petersen/difflogic)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732) (NeurIPS 2024) — includes post-training logic synthesis
- [Light Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250) (2025) — depth via reparameterization (the backprop-side answer to the same problem)
- [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) (Hinton, 2022)
- Cascade-Correlation (Fahlman & Lebiere, 1990) — the original "grow and freeze" network
- Greedy layerwise learning can scale to ImageNet (Belilovsky et al., ICML 2019) — block-wise greedy training with auxiliary heads, the closest relative of `--window`
- To our knowledge, **the combination** (LGN × backprop-free layer-wise training × adaptive depth × incremental simplification) has no published precedent as of mid-2026. If you know of one, please open an issue.

## License

MIT

---

## 日本語概要

論理ゲートネットワーク(DLGN)を**逆伝播なしで1層ずつ**学習する実証実験です。各層をローカルな損失(GroupSum+交差エントロピー)で学習したら**即座に離散化して凍結**し、次の層は本物の0/1ビットの上で学習します。検証精度が頭打ちになったら層の追加を止めるため、深さは自動決定されます。学習後に回路を簡略化し、出力が完全に同一であることをビット単位で検証します。

結果は正直に言って一長一短です: 素のgreedyはend-to-end逆伝播に約5pt負けますが(88.2% vs 93.6%)、離散化ギャップが構造的にゼロ、学習メモリが深さ分の1、深さの自動決定という利点があります。その5ptの内訳を潰していくのが各実験です — **メモリ等価**(幅4倍でe2eと同じfloat予算)ではgreedyが3シード全勝(95.0% vs 91.5%)、**skip connections**(`--skip-input`)で初めて深さが精度に貢献、**MNIST**でも同じ構図が再現(84.6% vs 80.1%)、**先読み窓**(`--window 2`: 2層先まで逆伝播で共同学習してからまとめて離散化)で近視由来のギャップの約2/3を回収(90.4% vs 91.5%)、**アンサンブル投票**(`--ensemble M`: 独立学習した離散回路を横に並べて投票 — 並列評価なのでレイテンシ不変、投票回路込みで純粋な論理回路のまま)は他の全レバーと加算され、digitsで**96.4%**(自己ベスト)、MNISTでは4×500ゲートが単発2,000ゲートの従来ベストを半分の学習メモリで上回ります(84.7% vs 84.6%)。MNISTのスケーリングでは幅が支配的レバーで(4,000ゲート単発89.8%)、アンサンブル併用の**90.9%**で初めて90%を超えました(エポック増とwindow×幅は各+0.1ptで死にレバーと確認)。**Forward-Forward目的関数**(`--objective ff`: goodnessがバイナリ層ではpopcountになり、推論まで含めて純論理回路)は digits では教師ありローカルCEに2.4pt負けますが、**MNISTでは+2.5pt逆転**(76.8%)し、skipなしで初めて深さ17層を精度に変換しました(ランダム配線がラベルを見られるようにするラベルビット複製が必須という発見つき)。逆伝播は12層でチャンスレベルに崩壊する一方、greedyは40層目でも学習が成立します。

各実験のセットアップ・数値表・**反証された仮説**(オーバーラップコミットはブロック式に勝てない、skipはパススルーを減らさない、DenseNet式は僅かに劣る、など)は [RESULTS.md](RESULTS.md) に、生ログは実験ごとの個別issue(#1〜#7、RESULTS.mdの各セクションからリンク)にあります。

新規性の境界: 本リポジトリで唯一新しいのは「**各層を学習→即離散化→凍結し、次層を本物のビット上で学習する**」という一点です。離散化ギャップゼロはその直接の帰結、ハードウェア・イン・ザ・ループ成長はその派生(未実証)、メモリ効率と適応深さはCascade-Correlation / Forward-Forward由来の借り物です。詳細は「What is new here — and what is not」を参照してください。
