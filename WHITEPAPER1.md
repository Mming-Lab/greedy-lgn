# Backprop-free Logic Gate Networks — a reproducible experiment log

### Vol. 1: the climb to 94% on MNIST, and what each lever taught

> **日本語アブストラクト（要約）**
> 論理ゲートネットワーク（LGN）を、層をまたぐ逆伝播なしで1層ずつ学習する実証実験の記録です。各層をローカル損失で学習したら即座に0/1へ離散化して凍結し、次の層は本物のビット上で学習します。**500ゲート/層・単発ネットという固定予算**の下、素の88%から出発し、残差readout（＝素朴なブースティング）・skip配線・入力二値化の積み重ねで、**MNISTをビット等価検証済みで94.3%（3シード平均、単発ベスト94.6%）**まで到達しました。これは新手法の提案ではなく、**既存部品の組合せで制約下どこまで行けるかの探検記**です（勝ちも負けも正直に記録。新規性・優先権は主張しません。既出ならご指摘ください）。到達値は同予算のe2e逆伝播や、20倍以上のゲートを使うdifflogic（~97.7%）には及びません — その距離こそが次の章の題材です。

---

Logic Gate Networks ([difflogic](https://github.com/Felix-Petersen/difflogic), Petersen et al.) learn circuits of 2-input Boolean gates by relaxing gate choice to a softmax over the 16 two-input functions. Once discretized they run with no multipliers, no DSPs, and no floating point — a natural fit for FPGA LUTs. This document is a log of one question: **how far can you get if you refuse to backpropagate across layers at all?**

It is a playground log, not a paper. Nothing here is peer-reviewed; the only "literature search" was asking an AI assistant. I make **no novelty or priority claims** — most of these ideas certainly exist under names I don't know. What follows is honest about the wins *and* the losses, with every number reproducible from the repo.

## Why refuse backprop?

End-to-end backprop training of LGNs has three well-known pain points:

1. **Vanishing gradients in depth.** With the standard parameterization, gradient norms fall below machine precision after ~16 logic layers ([Light DLGN, 2025](https://arxiv.org/abs/2510.03250)).
2. **The discretization gap.** Networks are trained as soft mixtures of gates and hardened afterwards; the mismatch costs accuracy ([Mind the Gap, 2025](https://arxiv.org/abs/2506.07500)).
3. **Training memory.** Every gate holds 16 float logits, and backprop keeps the whole network soft at once.

The recipe explored here sidesteps all three by construction:

> *Train one logic layer with a local loss (GroupSum + cross-entropy), discretize it immediately, freeze it, and train the next layer on the real 0/1 bits. Stop adding layers when the accuracy plateaus. Then simplify the circuit and verify it is bit-identical.*

Because each frozen layer is hardened *before* the next one trains, later layers learn on genuine Boolean inputs and **the reported accuracy is the hard circuit's** — the discretization gap is structurally zero. Only one layer is ever soft, so float memory is `gates × 16`, independent of depth. Depth is chosen automatically. None of these properties are novel (see *What this borrows*); the interesting part is what happens to accuracy.

## The arena: 500 gates/layer, one network

Every experiment on the main track runs at a **fixed budget — 500 gates per layer, a single network** — because the game is watching which *ideas* move the accuracy, not how much compute is spent. Two datasets play two roles: **`sklearn` digits (8×8)** is the fast bench (seconds per run, deterministic on CPU) for direction-finding; **MNIST (28×28)** is the referee — claims are only trusted once MNIST agrees. When the two disagree, MNIST wins.

The honest starting line, on digits:

| | greedy (this repo) | end-to-end backprop |
|---|---|---|
| hard-circuit test accuracy | 88.2% | **93.6%** |
| discretization gap | **0 (by construction)** | 0.0 here, +8.2 pt at a smaller/undertrained config |
| float logits during training | **8,000 (one layer)** | 32,000 (×4) |

Plain local training loses ~5 pt to backprop. The rest of this log is about buying that back — and then some — without ever letting a gradient cross a frozen layer.

## The climb (MNIST, 500 gates/layer, single net)

| step | idea | MNIST hard test acc |
|---|---|---|
| 0 | plain greedy (GroupSum + CE) | 74.3% |
| 1 | + windowed lookahead (`--window 2`) | 76.6% |
| 2 | Forward-Forward objective + lookahead + wrong-answer review | 82.0% |
| 3 | **residual / boosting readout (`--group-residual`)** | **90.9%** |
| 4 | **+ skip-input wiring (`--skip-input`)** | **93.9%** |
| 5 | **+ low-plane input binarization (`--thresholds`)** | **94.27%** (3-seed mean; best single 94.60%) |

Steps 3–5 are the story. Each is bit-exactly verified through simplification (the hardened, simplified circuit is checked identical to the trained one). The final 94.27% is a 3-seed result (94.08 / 94.60 / 94.13), winning 3/3 over a same-protocol control.

**The clear winner is the residual readout.** Plain greedy throws away every layer's class prediction except the last — which is exactly why accuracy decays with depth: shallow layers see fresh image information, but their good answers are discarded. Accumulate each layer's prediction instead (plain boosting / deep supervision) and the decay vanishes: a single 500-gate net climbs from 74.3% to **90.9%**, matching a scaling-track flagship that needed 4,000-gate layers × 4 ensemble members (≈16× the gates plus voting) — at a fraction of the inference area. Stacking skip wiring (re-expose the image to every layer) compounds to 93.9%, and adding a lower binarization plane (see below) reaches 94.27%.

## What each lever taught (wins and losses)

- **Residual readout — the win.** The single lever that turns depth from a liability into an asset. It is plain boosting applied to a logic circuit; the contribution is only the observation that it fixes greedy-LGN's depth decay cleanly.

- **Identity warm-start (`--warm-start`) — retires the lookahead window, but is not a residual substitute.** Initialize each new layer to *reproduce the previous one* (a ResNet identity block in logic gates), then refine. On plain greedy it beats the lookahead window by +4 pt (digits 94.5 vs 90.4, 3 seeds) and keeps depth productive to 15 layers. But it lands below the residual readout and doesn't compound with it (they cure the same disease).

- **Recursion (`--recur`, `--seq`) — identity is the precondition, and its direction is the point.** Weight-tied recursion, either iterating a layer in place (`--recur`) or feeding an image row-by-row with a hidden state (`--seq`, following [RDDLGN, arXiv:2508.06097](https://arxiv.org/abs/2508.06097)). Both **collapse from random init and are rescued by identity warm-start** — and in `--seq` the identity must point at the *previous layer's* state, not the cell's own (self-reference collapses 82%→27%). This makes it three-for-three with RDDLGN's own finding (their Residual hidden-state init is essential, Gaussian collapses): **recursing discrete logic needs a near-identity map pointing at informative bits.**

- **Adaptive per-layer epochs — a documented negative.** Stopping each layer when its discrete circuit settles *should* save compute. It doesn't beat the fixed 120 epochs on accuracy — the argmax churn half-decays at ~125 epochs, so the default was already near-optimal. A useful by-product: fully settling a layer *stalls depth growth*, because a saturated layer's tiny gain trips the depth-stop. Per-layer convergence is not the objective.

- **Input binarization (`--thresholds`) — preprocessing, and the diagnosis was backwards.** A pixel census found 34/192 digit bits dead and the fixed thresholds below the pixel quantiles — suggesting *raise* them. The data said the opposite: quantile thresholds lose, and **adding a lower plane wins** (residual MNIST +0.75 pt, 3/3 seeds). Faint-stroke information below the old lowest threshold matters more than reviving dead bits. This is a data-representation lever, not a learning idea — it transfers to any method, so it's credited separately.

- **Convolution (`--local`, `--conv`) — in progress, memory-bound.** A locality prior alone (`--local`) is a **negative on MNIST** (−2.25 pt): without pooling the receptive field never spans the 28×28 image. Real convolution (`--conv`: weight-shared kernel trees + OR-pooling, following [conv-DLGN, NeurIPS 2024](https://arxiv.org/abs/2411.04732)) recovers on digits — level with the dense residual baseline where weight sharing has little to share — and pooling is load-bearing. The MNIST referee verdict is still pending: the soft-training tensor overflowed 6 GB until a memory rewrite (fold the 16 gates to a 4-term `{1,a,b,ab}` basis, gradient-checkpoint the tree, budget the hard-eval chunks) brought it in range. Whether convolution beats the residual champion on MNIST is the open question this log ends on.

## What the circuits actually look like (diagnostics)

Two standalone tools ([`diagnose.py`](diagnose.py), [`dynamics.py`](dynamics.py)) inspect the trained circuits:

- **Warm-start has a fingerprint.** Plain greedy spreads over the 8 simple gates (~9–11% each, constants ~0.5%); warm-start collapses the distribution onto gate A (pass-through, 39%; A-family ~70%). The identity bias is literally visible in the learned circuit, and it explains why warm layers are barely touched by training (62% keep their init) and simplify less (their pass-throughs are live).
- **~50% functional redundancy.** Half the gates produce an output column identical or complementary to an earlier gate — yet the simplifier's *structural* duplicate-merge finds zero (different wiring, same behaviour on the data). A quantified gap, honestly not the same as logical equivalence.
- **Recurrent cells are fading-memory maps, with real oscillators.** Rolling a trained `--seq` cell forward under blank input: the accuracy winner (warm-start) is a contractive map converging to a single fixed point in 2–4 steps — an echo-state / fading-memory regime. But plain `--seq` spontaneously grows period-2 and period-6 limit cycles (oscillators built from logic gates and a register), just on the losing configuration.

## Verified, and honest about the limits

Every flagship number is checked bit-exact: after training, a pure-Python pass folds constants, removes pass-throughs, merges structural duplicates, eliminates dead gates, and asserts the simplified circuit is identical to the trained one. The residual readout reads *all* layers, so the simplifier was extended to treat every layer as an output; the 94.27% champion verifies `identical = True` (the previously published 93.85% was unverified and is retired).

The limits, stated plainly:

- **Still behind backprop at equal config** (~5 pt on digits) and **far from difflogic's ~97.7%**, which uses >20× the gates. 94% here is "impressive *given the constraints*", not state of the art.
- **Convolution is memory-bound** on a 6 GB laptop GPU; the MNIST conv verdict is deferred, not delivered.
- **Single machine, small budgets.** No claim survives contact with a real scaling study.

The road to 97% — the line where this stops being a curiosity and starts being a plausible edge-inference story — runs through convolution done right (weight sharing + a receptive field that grows), which is exactly the unfinished thread above.

## What this borrows

Almost every ingredient is prior work; this section is about being explicit, not claiming credit.

| piece | where it comes from |
|---|---|
| Logic gate networks, FPGA-friendly inference | [difflogic](https://github.com/Felix-Petersen/difflogic) (Petersen et al., NeurIPS 2022) — the platform, not mine |
| Convolutional logic gates, post-training synthesis | [conv-DLGN](https://arxiv.org/abs/2411.04732) (NeurIPS 2024) |
| Recurrent logic gates, temporal state | [RDDLGN](https://arxiv.org/abs/2508.06097) (arXiv:2508.06097) — `--seq` follows its wiring |
| Residual readout | plain boosting / deep supervision (standard) |
| Grow-and-freeze, adaptive depth | Cascade-Correlation (Fahlman & Lebiere, 1990) |
| Local per-layer objectives | Forward-Forward (Hinton, 2022); block-wise greedy (Belilovsky et al., 2019) |

I have not surveyed the literature properly. If any of this — or the combination — duplicates prior work, that is expected; please open an issue so I can point to it.

## Reproducing this

```bash
pip install torch scikit-learn
python experiment.py --skip-e2e                                   # digits, plain greedy
python experiment.py --group-residual --skip-e2e                  # the winner
python experiment.py --dataset mnist --batch 512 --group-residual --skip-input --thresholds 31,63,127,191 --max-layers 40 --device cuda   # the 94% champion
python tests.py                                                   # regression: pinned, bit-exact
```

Full setups, number tables, and disproven hypotheses live in [RESULTS.md](docs/RESULTS.md) (scaling levers in [SCALING.md](docs/SCALING.md)); per-experiment raw logs are one GitHub issue each.

---

*Vol. 2, if it happens, is the convolution chapter — whether a growing receptive field closes the gap to 97%. No promises; the losses get logged either way.*
