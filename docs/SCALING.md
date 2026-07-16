# Scaling track (reference, off the main track)

The [README](../README.md) and [RESULTS.md](RESULTS.md) main track holds everything to a **fixed budget — 500 gates/layer, single network** — because the game there is which *ideas* move the accuracy points. This file collects the **scaling levers**: wider layers (`--gates`) and ensembles of independent nets (`--ensemble`). They spend compute and inference-circuit area, and reliably buy accuracy — but they don't tell you which idea is any good, so they sit outside the arena. This track is parked; it gets revisited only when a fixed-budget winner deserves a one-off scale check.

For reference, where the scaling levers take the same pipeline:

| | plain greedy (start) | best with scaling | how |
|---|---|---|---|
| **digits** | 88.2% | **96.4%** | 2,000 gates + `--skip-input`, ×4 ensemble (majority vote) |
| **MNIST** | 74.3% | **90.9%** | 4,000 gates + `--skip-input`, ×4 ensemble (soft vote) |

End-to-end backprop at equal *training memory* averages 91.5% on digits — the scaled stack is above it, at the cost of more inference area.

Note: the residual/boosting readout (a **fixed-budget** idea, see [RESULTS.md](RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes)) reaches **90.9% MNIST as a single 500-gate net**, matching this whole scaling stack at a fraction of the area — and it keeps scaling with width (2,000 gates → 95.4%). So the scaling flagship is no longer the repo's best number; it is kept here as the reference it always was.

## Detailed results

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
  *under the same budget*. Closing the absolute gap — wider layers, more epochs — is
  future work.
- Runtime: ~13 min for the 2,000-gate greedy run on an RTX 3060 Laptop. CPU would take
  hours; `digits` remains the CPU-friendly configuration.

Full run log: [issue #6](https://github.com/Mming-Lab/greedy-lgn/issues/6).


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
| 500 gates, plain ×4 | 87.7% | 91.1%¹ | 90.2% |
| 500 gates, plain ×8 | 87.6% | 92.0%¹ | 91.6% |
| 500 gates, W=2 blocks ×4 | 89.9% | 92.2% | **92.4%** |
| 2,000 gates + skip ×4 | 95.5% | 96.2% | **96.4% — repo best** |

¹ Corrected. The originally posted soft votes (91.3% / 91.8%) came from summing
τ-divided float counts across members, whose argmax flips on exactly-tied classes
depending on the device's floating-point reduction order — a CPU/GPU mismatch found
by the pinned regression suite (`tests.py`). Voting now sums exact integer counts
(mathematically the same argmax, deterministic on every device); members and
majority votes were unaffected. Corrected values are the deterministic ones.

MNIST (500 gates/layer, `--batch 4096 --epochs 30`, seeds 1–4; soft votes are the
corrected integer-count values, see footnote 1):

| config | member mean | soft vote | majority vote |
|---|---|---|---|
| plain ×4 | 74.5% | **83.0%** | 82.0% |
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
| 2,000 gates ×4 ensemble, soft vote | 87.3%¹ | ~15 min |
| 2,000 gates + W=2 blocks (single) | 84.7% | ~5 min |
| 4,000 gates (single) | 89.8% (depth 7) | ~8 min |
| 4,000 gates, 2× epochs (single) | 89.9% | ~15 min |
| **4,000 gates ×4 ensemble, soft vote** | **90.9%¹** | ~28 min |

¹ Ensemble soft votes are the corrected exact-integer-count values (see the
[ensemble-voting footnote](#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever));
the originally posted figures were 87.34% / 90.86%, off by ≤0.05 pt.

Findings:

1. **Width is the dominant lever, and it has not saturated.** Doubling width
   2,000 → 4,000 buys +5.2 pt for a single network (84.6% → 89.8%) — far more than any
   other lever at this scale. Simplification keeps ~42–45% of gates (28,000 → 12,517
   at 4,000 gates), bit-exact as always.
2. **Ensembling stacks at every width, with diminishing returns as members
   strengthen**: the ×4 vote adds +8.5 pt at 500 gates, +3.2 pt at 2,000, +1.3 pt at
   4,000. Combined best: **90.90%** (4 × 4,000 + skip, soft vote) — the repo's first
   crossing of 90% on MNIST.
3. **Two levers confirmed dead at this scale, reported honestly**: doubling epochs
   adds +0.1 pt (89.77% → 89.89%), and windowed lookahead on top of width+skip adds
   +0.1 pt (84.57% → 84.65% at 2,000 gates) — consistent with the digits finding that
   the window does not stack with skip. The myopia deficit appears to be whatever
   width and skip have not already fixed.
4. **Honest positioning unchanged in kind, narrowed in degree**: the gap to
   difflogic-scale results (~97.7%) shrinks from ~13 pt to ~7 pt, still with far
   smaller budgets (≤4,000 gates/layer, 30 epochs, single machine). Remaining known
   levers: 8,000-gate layers (see below) and convolutional wiring.
5. **8,000 gates: OOM at 6 GB, but the partial result is telling.** Before crashing
   while building layer 3, the 8,000-gate run reached **90.01% at depth 2** — above
   the best completed single 4,000-gate net (89.77% at depth 7), so width is still
   not saturated. The chunk budgeting fixed only the evaluation temporary; the real
   constraint is the **persistent wiring pools** (`[60000, 10352]` float32 plus
   transient copies during the skip-pool transition ≈ 7 GB). Fix candidates for a
   follow-up: uint8 pools (hard bits are bits), CPU-resident pools with per-batch
   transfer, or an in-place pool buffer.
6. **Width also lifts the residual readout (reference only).** The residual/boosting
   readout is a fixed-budget (500-gate) idea, but for reference, giving it more width
   scales cleanly on MNIST single net, no ensemble: 500 → **90.9%**, 1,000 → **93.3%**,
   2,000 → **95.4%** (depth 8). So a single 2,000-gate residual net already beats the
   old width×ensemble flagship (90.9%) by +4.5 pt. This is the scaling track, not the
   arena — the headline claims stay at 500 gates — but it shows the residual idea does
   not stop paying off when you spend area on it.

Full run log: [issue #9](https://github.com/Mming-Lab/greedy-lgn/issues/9).

## Ensembling saved checkpoints: members you already trained, for the price of inference

`--checkpoint` (added for the long depth-exploration runs) saves the *blueprint* of
every frozen layer — wiring plus gate selection. That means a finished run's `.pt` can
be rebuilt and voted with **no retraining at all**, which makes an ensemble out of runs
that were done anyway (`tools/vote_checkpoints.py`; members are `(.pt, depth)` pairs).
Using the three champion seeds from the depth-exploration batch (MNIST, residual+skip,
500 gates, each at its own selected depth):

| members | member agreement | best member | soft vote | vs best member |
|---|---|---|---|---|
| 2 seeds (s1@41 + s2@78) | 94.9% | 94.96% | **95.44%** | +0.48 |
| **3 seeds (+ s3@43)** | **92.5%** | 94.96% | **95.61%** | **+0.65** |
| 3 depths of seed 1 (@41/46/51) | 96.3% | 94.29% | 94.47% | +0.18 |
| 3 depths of seed 2 (@70/74/78) | 97.1% | 94.96% | 94.96% | +0.00 |

Two readings. **Different seeds work** — 95.61% off three runs that already existed,
for inference only (the arena headline is the 94.53% 3-seed *mean*; this is a different
measurement of the same runs, and it costs 3× the circuit area, so it stays off-arena).
**Same-seed depth snapshots do not.** With a residual readout, depth *d+k* is literally
depth *d* plus *k* more corrections, so the "members" are nested prefixes and can't
disagree — the agreement rate says it plainly (92.5% across seeds vs 96.3–97.1% across
depths of one seed), and the vote ties its best member outright on seed 2. Two diverse
members beat three nested ones. (Voting sums exact integer counts, per the issue #8
tie-break lesson; accuracies here are CPU-side and sit ~0.05 pt off the CUDA training
probes for the reason in [RESULTS.md](RESULTS.md)'s numerical footnote.)

### What the vote actually buys: latency, not compute

Because a checkpoint holds every committed layer, the same three seeds can be voted
**truncated at any depth**, for free. That answers a natural question — if you are
going to ensemble anyway, can you get away with shallower nets? — with no retraining.
Per-layer training cost is ~constant here (with `--skip-input` the wiring pool is
`2352 input + 500 gate` bits regardless of depth; seed 2 measured 343 s/layer), so
"layers trained" is a fair compute axis:

| depth *d* | 1 net (3-seed mean) | 3-seed vote | vote gain | layers trained | latency |
|---|---|---|---|---|---|
| 5 | 85.34% | 88.03% | +2.69 | 15 | 5 |
| 14 | 91.50% | 92.58% | +1.08 | 42 | 14 |
| 20 | 92.61% | 93.77% | +1.16 | 60 | 20 |
| 40 | 93.83% | 94.86% | +1.03 | 120 | 40 |
| 51 | 94.15% | **95.33%** | +1.18 | 153 | 51 |

The vote gain is strikingly **flat with depth** (~+1.0–1.2 pt from depth 8 on), so
voting and depth are not fixing the same errors. But budget-matched, the answer
reverses:

- **Equal training compute (42 layers):** one net at depth 42 ≈ 93.9% vs three nets
  at depth 14 = 92.58% — **depth wins by ~1.3 pt**.
- **Equal inference area (~21k gates):** same comparison, same conclusion.
- **Equal latency (depth 40):** one net 93.8% vs three nets **94.86%** — the
  ensemble wins by +1.0 pt.

So ensembling is *not* a way to trade layers for members: given a layer budget,
spend it on depth. What it buys is **critical-path latency** — three nets at depth
40 reach 94.86% against a single net's 94.91% at depth 78: about the same accuracy
at **half the latency**, for 1.5× the gates. On an FPGA target where the critical
path binds and area does not, that is a real trade; on a compute or area budget it
is not.

**More seeds is a poor use of the same compute.** Averaged over all subsets of the
three checkpoints:

| members | depth 20 | depth 40 | depth 51 |
|---|---|---|---|
| 1 → 2 | +0.90 | +0.90 | +0.91 |
| 2 → 3 | +0.26 | +0.13 | +0.28 |

The first extra member is worth ~+0.9 pt at every depth; the second only ~+0.2 pt —
decaying faster than the ~1/M of independent errors. Extrapolated, three more seeds
at depth 40 (120 layers ≈ 11.4 h) would buy perhaps +0.2 pt, while spending 33
layers (≈3.1 h) on depth 40 → 51 buys **+0.47 pt** — roughly 10× the compute
efficiency. (Honest limit: at k=3 there is only one subset, so the 2→3 figure is a
single draw, not an average; the decaying *direction* is consistent across all three
depths and is what theory expects, but the magnitude is soft.)

**The ceiling is shared, not individual.** All three members get 257 of the 10,000
test samples (2.57%) wrong *together*, so no vote of these members can exceed
**97.43%** — hard samples are hard for everyone. The vote reaches 95.61%, leaving
1.82 pt on the table: 207 samples have exactly one member right, and a majority
cannot reach them (soft-vote's score summing rescues 69). That oracle ceiling rises
with depth (96.18% @20 → 97.03% @40 → 97.29% @51), which is another way of saying
depth, not membership, is what buys headroom.

Full run logs: [issue #3](https://github.com/Mming-Lab/greedy-lgn/issues/3) (memory-matched), [#8](https://github.com/Mming-Lab/greedy-lgn/issues/8) (ensemble), [#9](https://github.com/Mming-Lab/greedy-lgn/issues/9) (MNIST scaling), [#14](https://github.com/Mming-Lab/greedy-lgn/issues/14) (checkpoint vote).
