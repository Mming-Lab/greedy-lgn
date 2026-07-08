# Detailed experimental results

Full write-ups of every experiment in this repo. The [README](README.md) carries the
headline numbers; this file carries the setups, tables, caveats, and the hypotheses
that did *not* survive contact with the data. Full run logs (environment, commands,
raw output) live in **one GitHub issue per experiment**, linked at the end of each
section below.

All experiments: `sklearn` digits (8×8, thermometer-binarized to 192 bits) unless
stated otherwise, 500 gates/layer, seed 1 unless a seed list is given. GPU runs on an
RTX 3060 Laptop (6 GB); CPU and CUDA results are bit-identical.

## Main comparison: greedy vs end-to-end backprop

| | greedy (this repo) | end-to-end backprop |
|---|---|---|
| depth | **4 (chosen automatically)** | 4 (copied from greedy) |
| hard-circuit test accuracy | 88.2% | **93.6%** |
| discretization gap | **0 (by construction)** | 0.0 at this scale¹ |
| float logits held during training | **8,000 (one layer)** | 32,000 (×4) |
| circuit after simplification | 2,000 → **1,316 gates (65.8%)**, bit-identical | — |

¹ With a smaller/undertrained config (`--gates 200 --epochs 30`), the end-to-end
baseline shows a **+8.2 pt discretization gap** while greedy remains at exactly 0. At
convergence on this easy dataset the gap closes; literature reports it re-appearing at
larger depth/scale.

**The takeaway is mixed, and that's the point.** Local training loses ~5 pt of accuracy
to backprop at this configuration — consistent with the Forward-Forward literature. In
exchange you get zero discretization gap, ~1/depth training memory, automatic depth,
and a circuit you can simplify incrementally. (See the windowed-lookahead section below
for how much of that 5 pt turned out to be myopia rather than anything fundamental.)

Also observed: **duplicate-gate merging found 0 duplicates** — with fixed random
wiring, two gates almost never share both inputs. The real simplification wins are
pass-through and dead-gate removal (34% of gates here). If you came for De Morgan-style
rewriting, this is the empirical answer.

Full run log: [issue #1](https://github.com/Mming-Lab/greedy-lgn/issues/1).

## Depth stress test: greedy survives 40 layers, backprop dies at ~12

We force greedy training to grow 40 layers (`--max-layers 40 --patience 40`) and train
end-to-end baselines at fixed depths (`--e2e-depth N`). Hard-circuit test accuracy:

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

1. **End-to-end backprop collapses to chance between depth 8 and 12** and never
   recovers. This is vanishing gradients, not undertraining: quadrupling the training
   budget at depth 40 (1,200 epochs) still gives 10.2%. Consistent with
   [Light DLGN](https://arxiv.org/abs/2510.03250)'s report of gradient norms below
   machine precision by ~16 layers — our layers are narrower (500 gates), which
   plausibly moves the cliff earlier.
2. **Greedy training never stops learning.** At layer 40 the local objective still
   reaches 69.9% train / 56.0% test. No gradient ever crosses a layer boundary, so
   there is no depth at which the training signal can die.
3. **Caveat: surviving depth ≠ exploiting depth.** Greedy's accuracy peaks at depth 4
   and decays monotonically afterwards — each additional hard layer loses information
   (no skip connections in this run). This experiment supports "greedy can train at any
   depth", not "deeper greedy networks are better".

Full run log: [issue #2](https://github.com/Mming-Lab/greedy-lgn/issues/2).

## Memory-matched comparison: equal training memory, greedy wins

Greedy's training-memory advantage (only one layer is ever soft) can be spent on width
instead. With 4× wider layers (2,000 gates), greedy holds the same 32,000 float logits
during training as the 4-layer end-to-end baseline:

| config | float logits during training | hard-circuit test acc (seeds 1/2/3) | mean |
|---|---|---|---|
| greedy, 500 gates/layer | 8,000 | 88.2 / 88.0 / 88.9 | 88.4% |
| greedy, 1,000 gates/layer | 16,000 | 92.7 (seed 1 only) | — |
| **greedy, 2,000 gates/layer** | **32,000** | **94.7 / 95.3 / 94.9** | **95.0%** |
| end-to-end, 500 × 4 layers | 32,000 | 93.6 / 90.4 / 90.4 | 91.5% |

- **At equal training memory, greedy beats end-to-end on every seed tested** (mean
  +3.5 pt) and with much lower variance (0.6 pt spread vs 3.2 pt). Depth is still
  chosen automatically (4 on all seeds) and the discretization gap is still
  structurally zero, while e2e shows small seed-dependent gaps (e.g. +0.9 pt on seed 2).
- **The honest cost: a larger inference circuit.** The memory-matched greedy circuit is
  ~5,300 gates after simplification vs 2,000 (raw) for e2e — greedy trades hardware
  area for training memory and cross-seed stability. (The simplification pass currently
  runs only in the greedy pipeline, so the e2e count is unsimplified.)
- Same toy-scale caveats as everywhere in this file: one easy dataset, 450 test
  samples, 3 seeds.

Full run log: [issue #3](https://github.com/Mming-Lab/greedy-lgn/issues/3).

## Skip connections: re-exposing the input turns survivable depth into usable depth

Classic residual addition (`x + f(x)`) does not exist in Boolean circuits, but its
cheapest circuit-native analogue does: with `--skip-input`, every layer's random wiring
pool becomes `[input bits ∥ previous layer]` instead of the previous layer alone. Zero
extra gates — it is only wiring. This directly attacks the information loss that made
greedy accuracy decay with depth:

| depth | greedy, no skip | greedy, `--skip-input` |
|---|---|---|
| 4 | **88.2%** (peak) | 87.1% |
| 8 | 84.9% | **90.4% (peak)** |
| 12 | 82.4% | 90.0% |
| 20 | 74.0% | 88.4% |
| 30 | 61.8% | 86.2% |
| 40 | 56.0% | 83.6% |

(500 gates/layer, growth forced to 40 layers, seed 1.)

- **The depth decay is largely gone** (layer 40: 56.0% → 83.6%), and for the first time
  **depth actually helps**: the peak moves from 88.2% at depth 4 to 90.4% at depth 8
  (+2.2 pt). The depth-stress-test caveat above — "surviving depth ≠ exploiting depth"
  — is now half-answered.
- **Combined with the memory-matched width** (2,000 gates/layer): **95.6 / 95.6 / 96.0%
  over seeds 1/2/3, mean 95.7%** — the best result in this repo, vs 91.5% mean for
  end-to-end at equal training memory. Depth is still chosen automatically (7/5/3), gap
  still structurally zero, simplification still verified bit-exact.
- **A hypothesis that did *not* survive the data**: we expected skip wiring to free the
  ~20% of gates that simplification reveals as pass-throughs (gates that only copy bits
  forward). The pass-through fraction stayed at ~20% with skip enabled. The benefit is
  information access, not gate savings — though skip circuits do simplify harder
  overall (47.8% of gates kept vs 65.8% without skip at the respective peaks).
- The e2e baseline is unchanged by this flag (standard DLGN wiring).

**DenseNet-style variant (`--skip-all`, negative result reported for honesty):**
exposing *all* previous layers (not just the input) gives the flattest depth curve of
all — 88.4% at layer 40, best 89.8% at depth 29 — but never beats `--skip-input`'s peak
(90.4%), and at memory-matched width it is slightly *worse* (95.1% vs 95.7% mean over 3
seeds), plausibly because the ever-growing pool dilutes the random wiring. One striking
side effect: dense circuits simplify dramatically harder — the 40-layer network shrinks
to **23.8%** of its gates (14,500 → 3,457, mostly dead-gate elimination), since later
layers cherry-pick the useful bits of the whole history. `--skip-input` remains the
recommended configuration.

Full run logs: [issue #4](https://github.com/Mming-Lab/greedy-lgn/issues/4)
(`--skip-input`), [issue #5](https://github.com/Mming-Lab/greedy-lgn/issues/5)
(`--skip-all`).

## MNIST: the pattern replicates (first pass, small budget)

Ported via `--dataset mnist` (28×28 → 3-threshold thermometer → 2,352 bits, standard
60k/10k split) with `--batch` minibatch training — full-batch training does not fit a
6 GB GPU at this scale; defaults remain full-batch and bit-identical for digits.

| config | float logits during training | hard-circuit test acc |
|---|---|---|
| greedy, 500 gates/layer, no skip | 8,000 | 74.3% (depth 6) |
| end-to-end, 500 × 4 layers | 32,000 | 80.1% (gap +0.1 pt) |
| **greedy, 2,000 gates/layer + `--skip-input`** | **32,000** | **84.6% (depth 9)** |

- **The digits-scale findings replicate on a 45× larger dataset**: at equal training
  memory, greedy + skip beats end-to-end by +4.5 pt, depth is chosen automatically, the
  discretization gap is structurally zero, and simplification removes 73% of the gates
  (18,000 → 4,814, verified bit-exact).
- **Honest positioning: these absolute numbers are far below the difflogic literature**
  (~97.7% on MNIST, using tens of thousands of gates per layer and much larger training
  budgets). This is a deliberately small-budget first pass (≤2,000 gates/layer, 30
  epochs/layer, single seed) where the meaningful comparison is greedy vs end-to-end
  *under the same budget*. Closing the absolute gap — wider layers, more epochs, better
  input binarization — is future work.
- Runtime: ~13 min for the 2,000-gate greedy run on an RTX 3060 Laptop. CPU would take
  hours; `digits` remains the CPU-friendly configuration.

Full run log: [issue #6](https://github.com/Mming-Lab/greedy-lgn/issues/6).

## Windowed lookahead: training two layers ahead closes most of the myopia gap

Plain greedy is myopic: each layer optimizes its own local loss with no knowledge that
more layers will follow. `--window W --commit J` interpolates between greedy and
end-to-end: train **W fresh soft layers jointly with backprop** on top of the frozen
prefix, then discretize and freeze only the **first J** of them and slide the window
(receding-horizon control is the closest analogy). `W=1` *is* plain greedy — the
refactored code reproduces the original results bit-exactly — and `W=depth` would be
end-to-end. `--win-loss` picks the window training loss: `last` (CE at the last window
layer only, pure lookahead) or `all` (CE averaged over all window layers, deep
supervision).

Digits, 500 gates/layer, 3 seeds, `--win-loss all` unless noted:

| config | hard test acc (seeds 1/2/3) | mean |
|---|---|---|
| plain greedy (W=1) | 88.2 / 88.0 / 88.9 | 88.4% |
| plain greedy, 2× epochs (control) | 86.7 / 89.1 / 88.7 | 88.2% |
| W=2 commit 1 (overlap / receding horizon) | 90.0 / 90.4 / 89.3 | 89.9% |
| **W=2 commit 2 (non-overlapping block)** | **90.4 / 90.9 / 90.0** | **90.4%** |
| W=4 commit 1 | 88.4 / 90.4 / 90.2 | 89.7% |
| end-to-end, depth 4 (discretized) | 93.6 / 90.4 / 90.4 | 91.5% |

Findings, honest parts first:

1. **`--win-loss last` collapses (42% on seed 1).** With the loss only at the end of
   the window, the committed layer is always trained as an *intermediate* layer — its
   own GroupSum readout, which is both the depth-selection probe and ultimately the
   circuit output, is never an objective. Deep supervision (`--win-loss all`) is
   required, not optional.
2. **Lookahead works, and it is not extra compute in disguise**: +2.1 pt mean over
   plain greedy, winning on all 3 seeds, while simply doubling plain greedy's epochs
   changes nothing (88.2%). W=2 block also costs roughly the same wall-clock as plain
   greedy (same epochs per committed layer).
3. **Most of the myopia gap closes**: greedy's 3-seed deficit vs end-to-end shrinks
   from 3.1 pt to 1.0 pt. Read as a decomposition of the flagship "greedy loses ~5 pt
   (seed 1)": roughly two thirds of the deficit was myopia, and one layer of lookahead
   recovers it.
4. **Overlapping commits lose to plain blocks.** The receding-horizon variant (commit 1
   of 2, re-planning every slide) was the motivating idea, but it never beats the
   simpler non-overlapping block (W=2 commit 2) while costing 2× the training. The gain
   comes from *training with lookahead*, not from re-planning.
5. **Wider windows don't help** (W=4: 89.7%). Two plausible mechanisms: deep
   supervision dilutes each layer's own objective (weight 1/W), and the co-adaptation
   built inside a soft window is destroyed at commit time — the larger the window, the
   more is destroyed.
6. **No stacking with `--skip-input`**: the best combination (W=2 commit 1 + skip,
   patience 4) reaches 89.7% mean vs 90.4% for windowless-skip-free W=2 blocks; skip
   alone at the same patience reaches 89.1%. The two mechanisms appear to address the
   same failure mode (information starvation of early commits) from different ends.
7. **What it costs conceptually.** For W>1 the flagship claim "no gradient ever crosses
   a layer boundary" weakens to "no gradient ever crosses a *frozen* boundary" —
   backprop is bounded to a constant-size window, so the vanishing-gradient immunity
   and the one-window training-memory bound are retained, and the committed prefix is
   still bit-exact with a structurally zero discretization gap. The default stays
   `--window 1`.

**MNIST confirmation** (500 gates/layer, no skip, `--batch 4096 --epochs 30
--max-layers 14`, seed 1): plain greedy 74.3% (depth 6) → **W=2 blocks 76.6% (depth
6), +2.4 pt**. The lookahead gain transfers to the 45× larger dataset. A visible
artifact: within each 2-layer block the first layer's probe is consistently weaker
(sawtooth curve) — the block's second layer is the one whose readout the window loss
optimizes most directly, and depth selection lands on even depths accordingly.

Full run log: [issue #7](https://github.com/Mming-Lab/greedy-lgn/issues/7).

## Ensemble voting: parallel circuits are the training-memory-free width lever

Hardware framing first: replicating a logic circuit M× costs area and power but **no
latency** (members evaluate in parallel), and a majority vote is itself a small Boolean
circuit — so an ensemble of hard networks plus its vote is still one pure logic
circuit, and each member's bit-exactness is verified independently by the
simplification pass.

`--ensemble M` trains M independent greedy networks that differ only in seed
(`seed .. seed+M-1`, which also randomizes each member's wiring) and reports two vote
rules:

- **soft vote**: sum the members' GroupSum counts, then argmax. Mathematically this
  equals concatenating the members' final layers into a single M×-wide GroupSum
  readout — i.e. the ensemble is a *block-diagonally wired* wide network.
- **majority vote**: per-member argmax, then plurality; ties broken by the summed
  counts (the tie-break is scaled so it can never overturn a vote lead).

Digits, seeds 1–4 (members 1–3 reproduce the known single-run numbers exactly):

| config | member mean | soft vote | majority vote |
|---|---|---|---|
| 500 gates, plain ×4 | 87.7% | 91.3% | 90.2% |
| 500 gates, plain ×8 | 87.6% | 91.8% | 91.6% |
| 500 gates, W=2 blocks ×4 | 89.9% | 92.2% | **92.4%** |
| 2,000 gates + skip ×4 | 95.5% | 96.2% | **96.4% — repo best** |

MNIST (500 gates/layer, `--batch 4096 --epochs 30`, seeds 1–4):

| config | member mean | soft vote | majority vote |
|---|---|---|---|
| plain ×4 | 74.5% | **82.9%** | 81.9% |
| W=2 blocks ×4 | 77.3% | **84.7%** | 83.9% |

Findings:

1. **Voting stacks with everything tried so far** — with windowed lookahead (92.4% on
   digits, beating the e2e 3-seed mean of 91.5% for the first time at 500 gates) and
   with skip+width (96.4%, new repo best). Contrast with window × skip, which did not
   stack: error decorrelation fixes a different failure mode than myopia or
   information starvation.
2. **The MNIST gain is much larger than the digits gain** (+8.4 pt vs +3.6 pt at 500
   gates): harder task, more room for members to disagree.
3. **New MNIST headline**: 4 × (500 gates, W=2 blocks) reaches **84.7%**, edging out
   the previous repo best of 84.6% (single 2,000-gate + skip) while holding **half the
   training memory** (2×500×16 = 16,000 float logits for the soft window vs 32,000)
   and fewer raw inference gates (13,000 vs 18,000). Members can also be trained in
   parallel on separate devices — greedy inside a member, embarrassingly parallel
   across members.
4. **Honest limit: ensembling is not a substitute for direct width.** At comparable
   inference area on digits, one 2,000-wide network (8,000 raw gates, 95.0% mean)
   clearly beats 4×500 members (5,500 raw gates, 91.3%). Joint training within a wide
   layer buys more than decorrelation across narrow ones. The ensemble's niche is
   converting inference area into accuracy **without touching training memory** — the
   exact lever the VRAM-bound MNIST scaling plan needs.
5. **Vote-rule crossover**: soft vote wins for weak members (it averages away
   overconfident mistakes), majority vote wins for strong members (one member's
   overconfident error can poison the summed counts but costs only one vote).

Full run log: [issue #8](https://github.com/Mming-Lab/greedy-lgn/issues/8).

## MNIST scaling: width × ensembles push past 90%

The absolute-accuracy follow-up to the MNIST first pass, testing the levers identified
above against each other. All runs: `--dataset mnist --skip-input --epochs 30
--max-layers 14`, RTX 3060 Laptop 6 GB. A code-level enabler shipped with this
experiment: `hard_batched` now scales its evaluation chunk inversely with layer width
(bit-exact, identical chunking for 500-gate configs), which keeps the `[B, G, 16]`
temporaries inside 6 GB at 4,000+ gates.

| config | hard test acc | runtime |
|---|---|---|
| 2,000 gates (first-pass best, for reference) | 84.6% | ~7 min |
| 2,000 gates ×4 ensemble, soft vote | 87.3% | ~15 min |
| 2,000 gates + W=2 blocks (single) | 84.7% | ~5 min |
| 4,000 gates (single) | 89.8% (depth 7) | ~8 min |
| 4,000 gates, 2× epochs (single) | 89.9% | ~15 min |
| **4,000 gates ×4 ensemble, soft vote** | **90.9%** | ~28 min |

Findings:

1. **Width is the dominant lever, and it has not saturated.** Doubling width
   2,000 → 4,000 buys +5.2 pt for a single network (84.6% → 89.8%) — far more than any
   other lever at this scale. Simplification keeps ~42–45% of gates (28,000 → 12,517
   at 4,000 gates), bit-exact as always.
2. **Ensembling stacks at every width, with diminishing returns as members
   strengthen**: the ×4 vote adds +8.4 pt at 500 gates, +3.2 pt at 2,000, +1.1 pt at
   4,000. Combined best: **90.86%** (4 × 4,000 + skip, soft vote) — the repo's first
   crossing of 90% on MNIST.
3. **Two levers confirmed dead at this scale, reported honestly**: doubling epochs
   adds +0.1 pt (89.77% → 89.89%), and windowed lookahead on top of width+skip adds
   +0.1 pt (84.57% → 84.65% at 2,000 gates) — consistent with the digits finding that
   the window does not stack with skip. The myopia deficit appears to be whatever
   width and skip have not already fixed.
4. **Honest positioning unchanged in kind, narrowed in degree**: the gap to
   difflogic-scale results (~97.7%) shrinks from ~13 pt to ~7 pt, still with far
   smaller budgets (≤4,000 gates/layer, 30 epochs, single machine). Remaining known
   levers: 8,000-gate layers (VRAM-feasible with the new chunk budgeting), better
   input binarization, and convolutional wiring.

Full run log: [issue #9](https://github.com/Mming-Lab/greedy-lgn/issues/9).
