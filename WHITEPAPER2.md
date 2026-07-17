# Backprop-free Logic Gate Networks — a reproducible experiment log

### Vol. 2: unlocking resources — three gates, one 6 GB laptop, still no cross-layer backprop

> **日本語アブストラクト（要約）**
> [Vol.1](WHITEPAPER1.md)は「500ゲート/層・単発ネット」という固定予算で、どのアイデアが精度を動かすかを見る本でした。この巻はその予算を外します。**幅と深さを解禁して、層をまたぐ逆伝播なしのままMNISTでどこまで登れるか**の記録です。結果: **97.04%**（3シード 97.00/97.08/97.05、8,000ゲート×深さ10、簡略化後約66,000ゲート、全ラン**ビット等価検証済み**、6GBのノートGPUで1本約40分）。第一関門（先行研究のコンパクト設計点95.8%）は幅を解禁した**18分後**に落ちました。ただし**ゲート効率では負けています** — 勾配学習の先行研究は同じMNISTを、difflogicが48,000ゲートで97.69%、LILogic Netが8,000ゲートで98.45%です。この巻の見どころは効率の勝利ではなく、**逆伝播を完全に捨てたレシピが97%台まで登れたこと**と、**どこで止まったか**（訓練データを暗記しきった地点で精度が飽和する壁）です。新規性・優先権は主張しません（文献調査は浅く、既出なら教えてください）。

---

Vol.1 ended at **94.64%** (3-seed mean) under a self-imposed budget: 500 gates per layer, one network — because the game there was watching which *ideas* moved the accuracy, not how much was spent. This volume moves past that budget. The recipe is unchanged — train one logic layer with a local loss, discretize it, freeze it, train the next layer on real 0/1 bits — but width and depth are now free.

**This is the next stage, not a broken rule.** The budget was never the method; it was a lens for comparing ideas, and it did that job. Vol.1's last finding was already pointing here: the champion had been stopping mid-climb because of a *stopping rule*, not a limit. Lift the budget too, and the question changes from "which idea wins?" to "**how far does the recipe actually go?**" (The lens isn't thrown away — new ideas still get measured at 500 gates, where a bad one dies in seconds. It just isn't the ceiling any more.)

The target: three gates borrowed from the literature.

| gate | line | who set it |
|---|---|---|
| 🟩 first | **95.8%** | a compact design point in the recent LGN literature (~18k gates) |
| 🟨 second | **97.0%** | roughly where lightweight binarized NNs (BNNs) land on MNIST — but they need XNOR+popcount, batch-norm, arithmetic |
| 🟥 third | **97.5%+** | [difflogic](https://arxiv.org/abs/2210.08277) territory (97.69% with 48k gates, e2e-trained) |

## The first gate fell in 18 minutes

One knob: `--gates 500` → `--gates 8000`.

| depth | test acc | wall clock |
|---|---|---|
| 1 | 87.2% | 3 min |
| 3 | 94.3% | 9 min |
| **5** | **96.2%** | **18 min** ← 🟩 first gate |
| 6 | 96.6% | 22 min |

Vol.1 spent 8–11 hours per seed to reach 94.6%. Width bought more than that in 18 minutes. That is not a subtle result, and it is worth being blunt about what it means: **the fixed budget, not the method, was the binding constraint on Vol.1's numbers.**

## The second gate: 97.04%, three seeds, verified

```bash
python experiment.py --dataset mnist --device cuda --group-residual --skip-input \
  --gates 8000 --batch 1024 --epochs 60 --max-layers 12 --patience 12 \
  --seed <S> --skip-e2e --checkpoint runs/vol2_8000_seed<S>.pt
```

| | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| **at depth 10** (uniform) | 97.00% | 97.08% | 97.05% | **97.04%** |
| gates (raw → simplified) | 80,000 → 66,225 | 80,000 → 66,307 | 80,000 → 66,100 | ~66k |
| bit-exact | ✅ | ✅ | ✅ | `identical = True` |
| best at own depth | 97.04% @18 | 97.15% @9 | 97.13% @12 | 97.11% |

**The headline is the depth-10 number: 97.04%**, not the 97.11% of per-seed bests. Two reasons, both about honesty. First, a uniform depth means the same protocol on every seed. Second, picking each seed's best depth means picking the best of ~12 draws from a noisy criterion — the probe is the *test set*, so more search means more optimism (Vol.1 measured this bias; it is real). Depth 10 costs 0.07 pt and buys a claim that does not depend on how long we searched.

Seed spread is **0.08 pt** — an order of magnitude tighter than Vol.1's 0.92 pt at 500 gates. Width appears to average out the luck of the wiring draw.

**🟨 second gate: cleared.** A pure logic circuit — no multipliers, no batch-norm, no popcount-and-threshold arithmetic, just AND/OR/XOR/… wired together — reaching the accuracy band of lightweight BNNs. The circuit is not a soft blend that gets rounded off at the end: every number above *is* the hard circuit's, verified gate-by-gate against the trained network.

## The wall: memorization, not capacity

Then it stopped. Hard.

| depth | train | test | |
|---|---|---|---|
| 8 | 99.86% | 96.90% | |
| **10** | **100.00%** | **97.00%** | training set fully memorized |
| 14 | 100.00% | 96.95% | |
| 18 | 100.00% | 97.04% | +8 layers bought **+0.04 pt** |

At depth 10 the network has memorized all 60,000 training images. After that, eight more layers — 64,000 more gates, 24 more minutes — bought four test samples. (Curiously, not *zero*: test creeps up 97.00 → 97.04 with train pinned at 100%. That is textbook boosting behaviour — margins keep widening after training error hits zero — and the residual readout *is* boosting. But at 8,000 gates a layer, it is not a trade worth making.)

Width scaling says the same thing from the other side:

| width | best test | raw gates at best |
|---|---|---|
| 500 (Vol.1) | 94.64% | 39.5k–58.5k |
| 2,000 | 96.61% | 80,000 |
| 8,000 | 97.11% | 72k–144k |

Each 4× in width buys less: the error shrinks by ~0.85× per doubling, far from the ~0.7× the early (500→2,000-gate) scaling law suggested. Extrapolating that law predicted **97.65%** at 8,000 gates; the measured 97.11% is **0.5 pt short**. There is a floor around 2–3% error that width alone is not touching.

**The honest reading: past a point, width buys memorization, not generalization.** That is the wall this volume ran into, and it is why the third gate stays shut.

## The efficiency fight: we lose, clearly

The second gate is an *absolute accuracy* claim. On **gates per point of accuracy**, the picture inverts:

| model | training | gates | MNIST |
|---|---|---|---|
| [LILogic Net](https://arxiv.org/abs/2511.12340) (2025) | gradient-based, learnable connectivity | **8,000** | **98.45%** (<5 min) |
| [difflogic](https://arxiv.org/abs/2210.08277) small (2022) | e2e backprop | 48,000 | 97.69% |
| difflogic large | e2e backprop | 384,000 | 98.47% |
| **this work** (8,000 × depth 10) | **layer-local only** | **~66,000** | **97.04%** |
| this work (2,000 × depth 40) | layer-local only | ~65,000 | 96.61% |

LILogic Net reaches **+1.4 pt on 8× fewer gates**. That is not a near miss; it is a different league of gate efficiency, and it comes from doing exactly what this project refuses to do — optimize the whole network at once (there, including the wiring itself).

We also tried to buy efficiency by reshaping the same budget, and failed:

| shape at ~18k gates | test | | shape at ~48k gates | test |
|---|---|---|---|---|
| 500 × 36 | 93.6% | | 500 × 96 | 95.2% |
| 2,000 × 9 | 94.5% | | 2,000 × 24 | 96.1% |
| 8,000 × 2 | 92.3% | | 8,000 × 6 | 96.6% |

The aspect ratio matters (too thin wastes gates on redundancy; too fat starves the boosting chain of correction steps), but **no shape we tried closes the gap to 18k/95.8%, let alone 8k/98.45%**. Greedy layer-local training appears to pay for its independence in gates. A plausible mechanism, stated as a hypothesis: each layer optimizes for *itself*, so it hoards information the later layers duplicate — the circuit diagnostics in Vol.1 measured ~50% functional redundancy, which fits.

## What the 6 GB actually constrained

The laptop is part of the story, not the scoreboard. What it cost:

- 8,000 gates: comfortable (batch 1024, ~2.5 GB, 3 min/layer)
- **16,000 gates: OOM — but not where you would think.** Training fits. The pool *transition* did not: the code built the next layer's wiring pool with `cat([X, h])`, which briefly holds the old pool, the new pool, and the layer's output at once (~3.2 GB of transient at that width). Fixed by allocating the pool once and writing the layer's output in place — under `--skip-input` the input half never changes. Same bytes, bit-exact (pinned by the regression suite), one fewer copy. 16,000 gates then trains fine; it is simply slow on this GPU (~7 min for layer 1, longer after, since the wiring pool grows to 18,352 bits).

Worth stating plainly: **the wall at 16k was our implementation, not the method or the hardware.** The 8,000-gate headline was never memory-bound. A 16,000-gate run is in flight; on the width trend above it should be worth perhaps +0.1–0.3 pt, and this document will be corrected if it says otherwise.

## Verified, and honest about the limits

Every number in this volume is the hard circuit's, checked bit-exact: a pure-Python pass folds constants, removes pass-throughs, merges structural duplicates, eliminates dead gates, and asserts the simplified circuit's output is identical to the trained network's, sample by sample. The residual readout reads all layers, so the simplifier treats every layer as an output. All three seeds: `identical = True`.

The limits:

- **97% on MNIST is not impressive in isolation.** A two-layer MLP does 98%. The interesting part is the *constraint* — no gradient crosses a layer boundary, ever — not the number.
- **Gate efficiency is a clear loss**, documented above. If your metric is silicon area per point, use LILogic Net.
- **Single machine, one dataset, three seeds.** MNIST is a solved problem and a forgiving one; nothing here says anything about CIFAR-10, which is exactly why it is next.
- **The depth probe is the test set.** Uniform depth 10 limits the damage but does not eliminate it. A validation split would settle it; not done here.

## What this borrows

Unchanged from [Vol.1](WHITEPAPER1.md#what-this-borrows): the platform is difflogic's, the residual readout is plain boosting, grow-and-freeze is Cascade-Correlation (1990), layer-local objectives are Forward-Forward and block-wise greedy. This volume adds no new mechanism at all — it turns two existing knobs (`--gates`, `--patience`) and reports what happened.

I have not surveyed the literature properly. **If reaching 97% on MNIST without cross-layer backprop is already known, I would genuinely like the reference** — please open an issue.

## Reproducing this

```bash
# the Vol.2 headline: 97.04% (3-seed mean), ~40 min/seed on a 6 GB GPU
python experiment.py --dataset mnist --device cuda --group-residual --skip-input \
  --gates 8000 --batch 1024 --epochs 60 --max-layers 12 --patience 12 \
  --seed 1 --skip-e2e --checkpoint runs/vol2_8000_seed1.pt

python tests.py     # regression: pinned, bit-exact
```

The three trained circuits are published as release assets ([`vol2-8000-checkpoints`](https://github.com/Mming-Lab/greedy-lgn/releases/tag/vol2-8000-checkpoints)) — rebuild them with `--checkpoint` or `tools/vote_checkpoints.py` and re-verify every number above without repeating the training. Raw logs: [issue #15](https://github.com/Mming-Lab/greedy-lgn/issues/15).

---

*Vol. 3, if it happens, is the structure chapter. Vol.2's wall was that width buys memorization; the question is whether **structure** — convolution done right, weight sharing with a receptive field that grows — buys generalization instead. MNIST's third gate (97.5%) is the warm-up; CIFAR-10 is the real test. No promises; the losses get logged either way.*
