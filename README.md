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

**The takeaway is mixed, and that's the point.** Local training currently loses ~5 pt of accuracy to backprop — consistent with the Forward-Forward literature. In exchange you get zero discretization gap, ~1/depth training memory, automatic depth, and a circuit you can simplify incrementally. Whether that trade is worth it is exactly what the roadmap below is for.

Also observed: **duplicate-gate merging found 0 duplicates** — with fixed random wiring, two gates almost never share both inputs. The real simplification wins are pass-through and dead-gate removal (34% of gates here). If you came for De Morgan-style rewriting, this is the empirical answer.

## Quick start

```bash
pip install torch scikit-learn
python experiment.py                                      # full run, a few min on CPU
python experiment.py --gates 200 --epochs 30 --max-layers 3   # ~20 s smoke test
python experiment.py --skip-e2e                           # greedy + simplification only
```

## Roadmap / open questions

- [ ] **Depth stress test**: can greedy training grow 30–40 layer LGNs where standard backprop gradients vanish? (This would be the headline result.)
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
