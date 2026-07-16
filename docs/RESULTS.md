# Detailed experimental results

Full write-ups of every experiment in this repo. The [README](../README.md) carries the
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

**MNIST baseline — same budget, and the story flips.** The digits table above uses one
depth (4) because that is where plain greedy peaks. On MNIST the honest e2e reference is
its *best* over depths, since e2e can't use much depth: a single 500-gate/layer e2e net
peaks at **81.76% @ depth 6** and then collapses (80.6 @4, 81.8 @6, 77.7 @8, 71.7 @10 —
vanishing gradients, `--e2e-depth` sweep, seed 1, `--batch 512`). Plain greedy on MNIST
is 74.3% — so at the plain starting line e2e wins by ~7 pt. But the residual lever lifts
greedy to 90.9% and the method headline (residual+skip) to **93.7%** (3-seed mean 93.72),
while e2e is stuck at 81.8% because it cannot go deep. Net: on the same single-500 budget,
**backprop-free greedy beats end-to-end backprop by +11.9 pt on MNIST** (93.7 vs 81.8),
up from +2.8 pt on digits (96.4 vs 93.6). (An off-arena preprocessing lever — a lower
input-binarization plane — adds +0.55 pt on top for 94.27%; see the input-binarization
section.) The mechanism is the depth-stress result below — e2e's gradients vanish
past ~6–8 layers; greedy has no cross-layer gradient to vanish.

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

## Forward-Forward objective: popcount goodness — behind on digits, ahead on MNIST

`--objective ff` swaps the per-layer local objective from GroupSum+CE to
Forward-Forward goodness (Hinton, 2022), keeping everything else in the pipeline
(train soft → discretize → freeze → adaptive depth → simplification with bit-exact
verification). The LGN-native observation that motivated this: on binary layers,
Hinton's goodness (sum of squared activations) degenerates to **popcount** — the
"is this input real?" detector is a single adder tree, and the full inference
procedure (overlay each of the 10 candidate labels, run the circuit, pick the label
with the highest popcount) remains one pure logic circuit. Within a layer, training
still uses gradients (as does the GroupSum objective); FF changes *what* the layer
optimizes, not the per-layer optimizer.

Mechanics: the one-hot label is concatenated onto the input bits (positive pass =
true label, negative pass = a random wrong label, resampled each layer); the loss
pushes positive goodness above θ = G/2 (the expected popcount at random init) and
negative goodness below it, scaled by √G.

**First honest failure: random 2-input wiring almost ignores 10 lone label bits.**
With a plain 10-bit overlay, ~90% of layer-1 gates never touch a label bit, so
positives and negatives look identical to most of the layer and FF barely beats
chance (18% on the smoke config). Hinton's dense layers see the label everywhere;
sparse random wiring does not. `--ff-label-rep K` replicates the label bits K times
so the wiring pool actually samples them.

digits, 500 gates/layer (GroupSum+CE reference: 88.4% mean over 3 seeds):

| config | goodness test acc |
|---|---|
| rep 8 + skip-input | 78.0% |
| rep 19 | 83.6% |
| rep 19 + skip-input | 80.2% |
| **rep 38 (seeds 1/2/3)** | **86.0 / 85.1 / 86.9 — mean 86.0%** |
| rep 57 | 83.6% |
| rep 76 | 82.2% |

Findings:

1. **FF lands 2.4 pt behind supervised local CE** (86.0% vs 88.4%, 3 seeds each) —
   closer than the typical FF-vs-backprop gaps in the literature, and with a readout
   that is nothing but a popcount comparison.
2. **The label-replication curve peaks at label:data ≈ 2:1** (rep 38 = 380 label bits
   vs 192 data bits) and falls off on both sides — under-replication starves gates of
   label access, over-replication starves them of data.
3. **`--skip-input` *hurts* FF** (83.6% → 80.2% at rep 19), the opposite of the
   GroupSum result. Not fully understood; noted as an open question rather than
   explained away.
4. **Adaptive depth stops itself at 6** — allowing 16 layers changes nothing (86.0%
   both ways). The Hinton-style normalization problem (later layers free-riding on
   earlier layers' goodness) does not visibly runaway here, plausibly because hard
   bits cap each gate's contribution at 1.
5. Simplification and bit-exact verification carry over unchanged (the verification
   input includes the overlaid label bits).
6. **MNIST reverses the verdict** (500 gates, `--batch 4096 --epochs 30`, rep 470 ≈
   the same 2:1 label:data ratio, seed 1): FF reaches **76.8% at depth 17** vs 74.3%
   (depth 6) for GroupSum+CE at the same width — **+2.5 pt in favour of FF** on the
   45× larger dataset, after losing by 2.4 pt on digits.
7. **FF is the first lever that exploits depth without skip wiring.** The probe
   climbs monotonically for 17 straight layers (35.4% → 76.8%) where no-skip GroupSum
   greedy peaked at depth 6 on MNIST (depth 4 on digits) and decayed. A speculative
   reading, stated as such: an FF layer's task — push positive popcount up, negative
   down — composes across layers (each layer refines an already-separated density
   code), whereas each GroupSum layer must re-encode class-count evidence from
   scratch and loses information at every discretization.
8. **Honest inference cost**: classification runs the circuit once per candidate
   label (10×, parallelizable in hardware area), and the readout is a popcount
   comparison instead of GroupSum count buckets.
9. **Windowed lookahead stacks with FF** (`--window 2 --commit 2`; the FF window
   loss is always deep-supervised, since each committed layer's goodness is the
   readout): digits 86.0% → **88.0%** mean over 3 seeds (+2.0 pt, all seeds up,
   nearly closing the gap to plain GroupSum's 88.4%), MNIST 76.8% → **78.2%** at
   depth 19 (+1.4 pt) — the best 500-gate single net on MNIST across all
   objectives. The contrast with window × skip (which did *not* stack) carries a
   mechanistic hint: FF's depth tolerance and lookahead fix different deficits,
   whereas skip and lookahead compete for the same one. `W=1` remains bit-identical
   to plain FF (regression-checked at 86.00%).

10. **Hard-negative mining (`--ff-neg`), motivated by how humans study — review what
    you got wrong**: instead of a uniformly random wrong label, mine the wrong label
    the frozen prefix currently finds most plausible, re-selected every time a layer
    freezes (a per-layer curriculum; layer 1 stays random). Costs nothing — the
    goodness matrix is already computed for the training probe. Three sub-findings:
    - **Pure hard negatives collapse** (32–65% on digits, dead by layer 1–2) — the
      negative distribution gets too narrow, consistent with hard-negative
      instability in the contrastive-learning literature.
    - **A 50/50 mix of hard and random works**: digits 86.0% → 88.1% (W=1, +2.1 pt)
      and 88.0% → **89.7%** on top of the W=2 window (90.0/88.0/91.1 over seeds) —
      the FF stack now *surpasses plain GroupSum* (88.4%) and closes on
      GroupSum×window (90.4%).
    - **No gain on MNIST from prefix-mined negatives** (77.9% vs 78.2% with random,
      within noise). At the time I read this as "MNIST has nothing to mine yet" — but
      finding 11 shows the real cause was mining *quality*, not MNIST.

11. **Warm-up before mining rescues hard negatives, and moves MNIST a lot
    (`--ff-neg-warmup`, `--ff-neg review`).** Two ideas from the same "how humans
    study" intuition: (a) don't mine from an untrained network — train each layer on
    random negatives for the first half of its epochs, *then* let the partly-trained
    layer itself sit the mock exam and mine its own mistakes; (b) `review` mode: the
    samples it got wrong study their own wrong answer, the ones it got right stay
    random (so the negative set never collapses to a few labels).
    - **Warm-up un-breaks pure hard negatives**: the collapse in finding 10 was mining
      from a layer that hadn't learned yet. With a 0.5 warm-up, pure `hard` on digits
      goes from 32% (dead at layer 1) to **88.0%** — a healthy run.
    - **Digits (at its ceiling) barely moves**: 3-seed means over the W=2 window are
      mix 89.7 / review-warmup 88.9 / hard-warmup 87.9 — plain prefix-mined mix still
      edges it. No new digits record.
    - **MNIST moves a lot**: `review` + 0.5 warm-up reaches **82.0% at depth 15**, up
      from 78.2% with random negatives (+3.8 pt) and from 77.9% with prefix-mined mix
      (+4.1 pt); pure `hard` + warm-up gets 80.5%. This is the best 500-gate single-net
      MNIST result in the repo, across every objective.
    - **The two-tier reading**: the earlier "MNIST is unmoved" was wrong — the mining
      was just low-quality (mining from a frozen prefix that is itself only ~75%
      accurate, several layers back). Letting the current layer grade its own mock exam
      fixes the quality, and MNIST responds strongly. This is also a clean example of
      why digits (near ceiling, insensitive) and MNIST (sensitive) can disagree: an
      idea that looks flat on digits can be a clear win on MNIST.

12. **Iterated mock exams and hard-sample boosting do NOT help — one exam is
    enough (`--ff-neg-phases`, `--ff-neg-boost`; hypothesis disproved).** Two
    natural extensions of finding 11's study analogy: re-take the mock exam
    several times per layer (re-mining fresher negatives each phase), and weight
    the loss of currently-misclassified samples higher. Both implemented (K=1 /
    B=1 reproduce finding 11 bit-exactly), both fail:
    - digits (3 seeds, on the review-warmup W=2 stack): all variants land in a
      flat 88.5–89.0% band — no signal, as expected at digits' ceiling.
    - MNIST (same stack, reference 82.03%): phases=3 → 81.3% (−0.7), boost=2 →
      81.0% (−1.0), both combined → **77.5% (−4.5, with a visibly unstable
      layer-to-layer curve)**.
    - Speculation, labelled as such: one exam already gives each mistaken sample
      its personally hardest negative, so re-mining mostly reshuffles the same
      set while cutting each phase's epochs; and boosting double-weights samples
      that `review` has already concentrated the negatives on. The clean version
      of the human-study analogy — warm up, one mock exam, review your own wrong
      answers — appears to be the whole effect.

13. **Structured wiring (`--ff-struct`) replaces the label-replication hack at equal accuracy — cleaner, not stronger.** digits (3 seeds, review-warmup W=2 stack, rep 1): f=0 collapses (66.4%, no label access), then monotone 0.25=82.4 / 0.5=86.7 / 0.75=88.7 / **1.0=90.3%**, beating the rep-38 force (88.9%) by +1.4 pt. At f=1.0/rep1 each gate detects one (feature, class) pair — a per-class feature bank the window's 2nd layer combines. **MNIST is a tie, not a win**: f=1.0 rep1 = 81.9% vs 82.0% for rep-470, within noise. The digits gain was a ceiling artifact. Verdict: structured wiring buys the same accuracy with a tiny pool (10 vs 4,700 label bits) and zero wasted gates, but is not itself an accuracy lever — its value is as the base for a repeating structured block (labels re-injected every block; future work).

Full run log and the window / negative-mining / warm-up / iterated-exam / structured-wiring follow-ups: [issue #10](https://github.com/Mming-Lab/greedy-lgn/issues/10).

## Residual/boosting readout: accumulate the answer, and the depth decay vanishes

The depth-decay diagnosis above (plain greedy peaks early, then decays without skip) has a cause worth naming: each layer produces its own GroupSum class scores, but **only the last layer's scores are used** — every earlier layer's answer is thrown away. The shallow layers (which still see fresh image information) make good predictions that are discarded; deep layers must re-derive the answer from information-degraded hard bits.

`--group-residual` keeps them instead: each layer's class scores are **added to the frozen layers' accumulated prediction** (each layer is trained to correct the running residual — plain gradient boosting / deep supervision). The prediction becomes `argmax` over classes of the **total class-c bits summed across all layers** — still a pure logic circuit (a bigger popcount), just reading every layer's output instead of only the last.

The effect is large, and it holds on the arbiter:

| | plain greedy (no skip) | residual |
|---|---|---|
| digits | 88.2% @4, decays to 76.9% @16 | climbs to **96.4% @10**, stable |
| **MNIST** | 74.3% @6, decays | climbs to **90.86% @9** (train 91.9 / test 90.6 — not overfitting) |

- **MNIST +16.5 pt over plain greedy, single 500-gate net, no skip, no window, no ensemble, in 96 s.** It *matches the previous repo MNIST best* (90.9%) — which needed 4,000-gate layers × 4 ensemble members (≈16× the gates plus voting). Residual reaches the scaled/ensemble flagship at a fraction of the inference area.
- **Not overfitting**: digits drove train to 100% (450 test samples), but MNIST's 60k training set keeps train (91.9%) and test (90.6%) close. The digits train=100% was a small-dataset artifact.
- **Why it works**: the plateau was never a supervision problem (each layer already sees the label) — the *answer* was discarded each layer. Accumulating it preserves the shallow layers' good predictions and lets deep layers only add corrections.
- **Honest limitations**: (1) the readout is plain boosting / deep supervision (standard ideas); the observation is only that it fixes greedy-LGN's depth decay cleanly. (2) The readout reads all layers' class bits (a larger popcount), not just the last. (3) `simplify` originally pruned gates not feeding the *last* layer and was skipped in residual mode; it now treats all layers as outputs and the residual circuits verify bit-exact (see the input-binarization section's champion run). ce loss, window=1 in this first version.
- Idea proposed by the project owner, from the intuition "learning is accumulation — so don't throw the answer away."

**Residual stacks with skip-input — new repo MNIST record.** Residual and skip both cure the depth decay, but by different mechanisms (residual accumulates the *answer*; skip re-exposes the *image*), so they compound:

| MNIST, 500-gate single net | acc | depth |
|---|---|---|
| residual (no skip) | 90.86% | 9 (converges fast, shallow) |
| residual + `--skip-input` (cap 25) | 93.12% | 24 |
| residual + `--skip-input` (cap 40) | **93.82%** (seed 1; 3-seed mean **93.72%** = headline) | 34 |

digits likewise: residual 96.4% → residual+skip **97.3%**. The headline is the 3-seed mean **93.72%** (verified; seed 1 shown above at 93.82% @34) — a single 500-gate net with no ensemble and no width scaling, it beats the previous repo best of 90.9% (which needed 4,000-gate layers × 4 ensemble ≈ 16× the gates plus voting). Honest cost: skip's +3 pt comes with a much deeper circuit (9 → ~30 layers = longer critical-path latency), so residual-alone (90.86% @9) is the shallow/fast champion and residual+skip (93.72%) is the max-accuracy champion — pick by whether latency or accuracy matters.

**Residual also revives the carry window (①+②).** Windowed lookahead was near-useless on plain greedy, and the carry-forward window (`--carry`, keeping the uncommitted lookahead layer instead of discarding it) actually *hurt* on its own (digits 90.2 vs 91.6). On top of residual, both flip positive: `--window 2 --commit 1 --win-loss all` adds +1.1 pt (MNIST 90.86 → 92.00), and adding `--carry` adds a further +0.25 pt (→ 92.25). The sign flip is the point — a carried layer's value in plain mode is its (fragile) features, which go stale when the layer below is frozen; in residual mode its value is the *answer correction* it contributes, which stays portable. So the project owner's "①+② should revive ②" was right in direction, though the gain is small and still below residual+skip (93.72%). Residual now supports `--window`; W=1 stays bit-exact.

Full run log: [issue #11](https://github.com/Mming-Lab/greedy-lgn/issues/11).

## Residual sample reweighting (`--group-boost`): a modest lever

AdaBoost-style reweighting on top of the residual readout: samples the frozen
running sum currently misclassifies get their CE weighted `B×` in the next
layer's training. The grader is the frozen prefix only (no self-grading), and
the residual already gives every misclassified sample a large gradient, so the
question was whether *explicit* reweighting adds anything.

| MNIST, 500-gate residual, seed | B=1 (plain residual) | B=3 |
|---|---|---|
| 1 | 90.14 | 91.04 |
| 2 | 90.06 | 89.94 |
| 3 | 89.67 | 90.18 |
| **mean** | **89.96** | **90.39 (+0.43)** |

digits 3-seed likewise: +0.67 pt at B=3. **Honest verdict:** a real but small
gain (2/3 seeds win, mean +0.43 pt on MNIST), consistent with the hypothesis
that gradient boosting already focuses on hard samples implicitly — the explicit
reweight only doubles down. Does not touch the residual+skip champion (93.72%, 3-seed mean).

## FF-Residual (`--objective ff-residual`): one-pass Forward-Forward on the residual readout — negative result

Hinton's Forward-Forward needs two passes (a positive image with the true label
overlaid, a negative image with a wrong label) because it trains all layers
"simultaneously" without backprop across layers. That two-pass trick is a
workaround for FF's original setting, not a requirement of the *goodness*
idea itself. In greedy-LGN each layer being trained is always the de facto
last layer (everything before it is frozen, its output feeds the readout
directly), so the same instruction — raise the true class's evidence, lower
the others — can be applied straight to the per-class scores, no label overlay,
one forward pass on the plain image.

Implementation (`--group-loss ffres`, `src/groupsum.py`): reuses the existing
residual accumulator (`--group-residual`'s running class-score sum) but
replaces its cross-entropy loss with a **per-class independent logistic loss**
— no softmax, so classes don't compete for probability mass. The accumulated
score is depth-centred (`Y_cum − d × gates/(2·n_class·τ)`, the expected score
of a random layer) so a fixed threshold at 0 stays meaningful as the sum grows
with depth; layers whose running total already clears the margin get near-zero
gradient from the softplus saturation (the "don't bother, previous layers
already did it" residual behaviour falls out automatically, no extra code).

digits 3-seed mean (500 gates, standard budget), vs the CE-residual champion:

| config | ffres | ffres + `--group-boost 3` | CE-residual champion |
|---|---|---|---|
| residual only | 95.41% | 95.92% (+0.51) | 96.4% |
| residual + skip | 96.29% | 95.93% (−0.36) | 97.3% |

**Honest verdict: loses to CE on both configs, consistently across all 3 seeds**
(not a 1-seed fluke — every seed underperforms the CE headline). Adding
`--group-boost` doesn't close the gap and its sign even flips between configs
(mixed, noise-level effect), so the loss isn't explained by missing explicit
hard-sample weighting. The likely reason: CE's softmax bakes in *implicit hard-negative
mining* — gradient on the wrong classes concentrates on whichever one is
currently most confusable, weighted by its softmax probability. The per-class
independent loss spreads gradient evenly over all 9 wrong classes instead,
which is a plausible way to lose ~1 pt (untested as a mechanism, only inferred
from the boost non-result). Checkpoint resume verified bit-exact for this
objective (it's a `groupsum` flag combination, no new state to persist).
MNIST untested — per this repo's digits/MNIST protocol a MNIST reversal isn't
ruled out, but a ~1 pt loss that boost didn't rescue lowered this lever's
priority rather than justifying the GPU time.

Full run log: [issue #12](https://github.com/Mming-Lab/greedy-lgn/issues/12).

## Identity warm-start (`--warm-start`): kills the lookahead window, not the residual

Instead of random init, each new layer (that has a previous layer) starts by
*reproducing the previous layer's output bits* — structured wiring `ia[i] =
(in_dim − gates) + i` into the pool tail (where the previous hidden bits live in
every skip mode) plus a logit bias `B` toward gate A (passthrough). The layer
begins as a non-destructive copy and learns the residual from there (a ResNet
identity block, in logic gates).

digits 3-seed (plain greedy, no residual):

| config | mean acc | note |
|---|---|---|
| plain | 88.37 | peaks at depth 4, then decays |
| plain + `--window 2 --commit 2` (lookahead) | 90.44 | |
| **plain + `--warm-start 5`** | **94.52** | climbs to the depth cap, never plateaus |
| residual (champion) | 96.30 | |

Warm-start **beats the lookahead window by +4.1 pt** and closes most of the
plain→residual gap — with no residual readout, just a better init. Raising the
depth cap, warm-start keeps climbing to depth 15 (0.9667) where plain greedy
decays after depth 4. On MNIST the direction replicates (warm-start beats the
window by ~+11 pt) but single-seed MNIST is noisy: the same config drew 0.8891
then 0.8504 on rerun (~4 pt CUDA nondeterminism, because accuracy is still
climbing steeply at the depth cap so early differences compound). With the cap
raised it climbs to 0.8707 @ depth 20 — **well below the residual champion
(0.9086 @ depth 9)**. So warm-start's real value is making *plain* greedy
depth-productive and retiring the lookahead window; it is not a residual
replacement. `residual + warm-start` is flat vs residual alone (like boost:
overlapping cures). Idea proposed by the project owner ("start each new layer
from an adjustable state, not random — maybe the window becomes unnecessary").

## Adaptive per-layer epochs (`--epoch-stop` / `--epoch-peak` / `--epoch-chain`): fixed 120 was already near-optimal

Replace the fixed 120 epochs/layer with early stopping on the **gate-argmax
change rate** (churn of the discrete circuit — smoother than the noisy hard
probe, and the only thing the next layer inherits). Four criteria were tried,
all calibrated on digits + warm-start:

- `--epoch-stop T` (settle): stop when churn < T. Needs `--epoch-min 70` — warm
  layers have a quiet churn *valley* around epoch 30-60 before ramping up, and a
  low min false-fires there.
- `--epoch-peak F` (weak-learner): stop once churn decays below F× the peak seen
  (commit half-baked layers, let depth work). F=0.5 was the best adaptive variant.
- `--epoch-peak-decay D`: depth schedule F·Dᐧ (deeper layers train longer).
- `--epoch-chain M`: layer 1 settles via `--epoch-stop`; its stop-time churn
  becomes the yardstick and each later layer stops at M× the previous layer's
  rate (auto-calibrated threshold). Project-owner idea.

digits + warm-start, 3-seed means: fixed-120 **95.6%** (±2.2), peak0.5 94.7%,
chain2 95.2% (±0.4), settle 94.2%, decay schedules 93–94% and shallower.
**Honest negative result:** no adaptive variant beats fixed 120 on accuracy.
Per-layer churn half-decays at ~125-130 epochs, so the default 120 already sits
near the optimum — adaptive mostly rediscovers it minus tuning noise. Two things
worth keeping: (1) `--epoch-chain 2` matches fixed-120 on the mean with **5×
lower variance** (a stability, not accuracy, win); (2) the mechanism revealed
that **fully settling a layer stalls depth growth** — a saturated layer's tiny
per-layer gain trips the depth patience, which also explains why the "saturate
the deeper layers" schedule backfired (auto-depth means saturating a layer
*causes* it to become the end of the stack). Per-layer convergence is not the
objective; slightly under-trained layers leave room for depth. MNIST untested.

## Within-layer recursion (`--recur`) and temporal recurrence (`--seq`): identity is the precondition, and its *direction* is the point

Two ways to recurse discrete logic, both weight-tied (learned logits stay one
layer's worth):

**`--recur K`** applies each layer K times to a static input (unrolled circuit K
layers deep per trained layer; `simplify` verifies the unrolled circuit
bit-exact). **`--seq`** presents the image one row per step (digits: T=8,
24 bits/step) and makes each layer a recurrent cell `s_t = L([x_t; s_{t-1}])`
trained by BPTT over T steps — wiring (concat) and GroupSum readout follow
RDDLGN ([arXiv:2508.06097](https://arxiv.org/abs/2508.06097)).
Frozen layers pass HARD state-bit sequences forward, so the
across-layer zero-gap property carries over (the within-layer temporal gap
remains, as in RDDLGN's 5.00→4.39 BLEU).

Both **collapse from random init** and are **rescued by identity warm-start**:

| digits seed 1 | plain (random init) | + identity warm-start |
|---|---|---|
| `--recur 3` | 0.836 @ depth 1 (collapse) | 0.938 @ 8 |
| `--seq` | 0.824 @ 3 | 0.909 @ 7 (warm 3) |

`--seq` 3-seed: warm3 **0.9118**, warm5 0.9111, plain 0.8296 — seeing only
24 bits/step, warm-start beats static plain greedy (0.884) by +2.8 pt, though
not static warm-start (0.945). A sharp mechanistic detail: in `--seq` the
identity must point at the **previous layer's** state (head of the concat), not
the layer's **own** state (tail) — tail-identity is a self-referential loop on a
zero-initialised state and collapses (82% → 27%). This makes the pattern
three-for-three: **recursing discrete logic needs a near-identity map, and it
must point at informative bits** — the same conclusion RDDLGN reaches from the
other side (their Table 7: "Residual" hidden-state init essential, Gaussian
collapses to 22.6%). MNIST `--seq` is deferred: 28-step BPTT over minibatches is
too slow for interactive turnaround on a 6 GB GPU (needs a multi-hour batch run).

## Input binarization (`--thresholds`): the diagnosis said "raise them", the data said "add a lower one"

The input has always been a fixed 3-plane thermometer (digits: pixel > 3/7/11,
MNIST: > 63/127/191). A quick input census (motivated by the circuit
diagnostics) found 34/192 digits bits **dead** (never fire — corner pixels that
never exceed even threshold 3) and the fixed thresholds sitting *below* the
nonzero-pixel quantiles (p25/p50/p75 ≈ 5/10/15) — suggesting quantile-aligned
thresholds should help. `--thresholds` makes the binarization configurable:
an absolute list (`--thresholds 5,10,15`) or train-set quantiles
(`--thresholds q4` = 4 planes, computed on the train split only). Default is
the original path, bit-identical.

**The diagnosis did not survive contact with the data.** Quantile-aligned
thresholds *lose* (digits plain 88.2 → 87.3; residual 96.0 → 95.1, seed 1), and
so does every variant that *raises* the lowest threshold. The only winner goes
the other way — **keep everything and add a lower plane** (digits `1,3,7,11,15`,
MNIST `31,63,127,191`): the faint-stroke information below the old lowest
threshold matters more than reviving dead bits.

| residual, 500 gates, 3 seeds | default | + low plane | Δ |
|---|---|---|---|
| digits (`1,3,7,11,15`) | 96.30% | 96.59% | +0.3 pt (2/3 seeds, ceiling) |
| **MNIST (`31,63,127,191`)** | **89.96%** | **90.71%** | **+0.75 pt (3/3 seeds)** |

MNIST is the referee and the gain *grows* there (90.64/90.73/90.76 — the new
config's seed spread is 0.12 pt vs 0.47 for the default, so it is also more
stable). Input bits cost no gates; the only price is a slightly wider layer-1
wiring pool. Honest caveat: the published residual-alone 90.86% was a single
draw from an earlier session — the claim here is the same-protocol 3-seed
comparison (89.96 → 90.71), not "beats 90.86".

**Stacked on the champion — off-arena preprocessing, verified, 3 seeds.** Stacking the
low plane on the champion (residual + `--skip-input`, cap 40), each seed run
side-by-side with a same-protocol control:

| residual + skip, MNIST | control (63,127,191) | + low plane (31,63,127,191) |
|---|---|---|
| seed 1 | 93.82% @34 | 94.08% @27 |
| seed 2 | 93.48% @26 | **94.60% @40** (still climbing at the cap) |
| seed 3 | 93.87% @29 | 94.13% @33 |
| **mean** | **93.72%** | **94.27%** (+0.55, **3/3 seeds**: +0.26/+1.12/+0.26) |
| bit-exact | identical = True (all) | identical = True (all) |

**The method headline is the control's 93.72% (3-seed mean); the low plane adds
+0.55 pt for 94.27% (best single 94.60%).** But it changes the *input encoding*
(a fourth threshold plane = more input bits), not the network or the learning, so
it is credited **off the arena** — not as the headline. Every run is verified
end-to-end: with simplify now supporting the residual all-layer readout, each
simplified circuit (~10.5–16k gates, 80–81% of raw) is confirmed bit-identical to
the trained network — the old unverified 93.85% is retired.
Honest correction: the seed-1 draw looked *shallower/smaller/faster* than its
control, but that did not hold across seeds (seed 2's low-plane run climbed to
the depth-40 cap and was still rising). So the robust claim is just the
accuracy — **+0.55 pt on 3/3 seeds**, consistent with the residual-alone
low-plane result (+0.75, 3/3) — not a depth or area win.

Full run log: [issue #13](https://github.com/Mming-Lab/greedy-lgn/issues/13).

## Convolutional wiring, phase 1 (`--local`): a locality prior alone just starves the net (negative)

First step toward convolutional wiring: keep everything else and only change
the wiring prior — every gate gets a pixel position and draws its two inputs
from the K×K neighbourhood of the pool (inputs at their pixel, previous gates
at their assigned position; gate positions random so they stay uncorrelated
with the GroupSum class groups, or inherited under `--warm-start`). No weight
sharing, no pooling.

digits said yes, the referee said no — the sharpest digits/MNIST reversal so
far (residual, seed 1): digits +0.4 pt (local3) / +0.7 pt (local5), but MNIST
**87.89 (−2.25)** at local3 and 89.90 (−0.24) at local5 vs 90.14 baseline. The
mechanism is clean: a 3×3 window covers 14% of an 8×8 image but 1.1% of 28×28,
and without pooling the receptive field grows only with depth (3×3 over the
residual's natural depth 8 ≈ 17×17 < 28×28) — no gate ever sees the whole
digit. Widening the window recovers most of the loss, confirming the reading.
**Locality alone is an information restriction**: it takes away the long-range
mixing that global random wiring provided for free, and gives nothing back.
The convolutional benefits (weight sharing = more effective samples per
parameter, pooling = growing receptive field) are exactly the parts this
phase left out — so this negative sharpens the phase-2 hypothesis rather than
killing the direction.

**Phase 2 (`--conv`): real convolution recovers, and pooling is load-bearing.**
`--conv C` builds C weight-shared kernels (a depth-`--conv-tree` gate tree whose
leaves wire inside a `--conv-k` window), replicated over all positions, then
`--conv-pool`×pool OR-pooled — following convolutional DLGNs (Petersen et al.
2024). digits calibration (seed 1, residual): plain conv starts at 0.46, the
residual readout rescues it (C64 0.90, C128/tree2 0.93), and **C128/tree3
reaches 0.9600 — level with the dense residual baseline** on an 8×8 field where
weight sharing has almost nothing to share. Removing pooling drops it to 0.86,
and k=5 also loses to k=3 — the phase-1 lesson (the receptive field must grow)
confirmed from the other side.

The seed-1 0.9600 above was a single draw of the heaviest config (C128/tree3,
45 min); the 3-seed picture (residual, digits) is more sober and puts conv
*behind* the dense baseline on this tiny field:

| digits, residual, 3 seeds | mean |
|---|---|
| dense residual (control) | **96.30%** (96.00/97.11/95.78) |
| conv C128/tree2 | 92.15% (92.00/91.56/92.89) |
| conv C64/tree3 | 90.67% (90.67/89.78/91.56) |

So conv trails dense by ~4 pt on 8×8 (expected — no receptive field to grow),
and **more channels beat a deeper tree** (C128/tree2 > C64/tree3, 3/3 seeds) —
which sets the default for the MNIST run. Memory was the real blocker and is
now fixed: folding the 16 gates to a 4-term `{1,a,b,ab}` basis, gradient-
checkpointing the tree, and budgeting the hard-eval chunks bring MNIST conv
within 6 GB (layer probes now appear where all three overnight configs
previously OOM'd). The MNIST referee verdict — does a grown receptive field let
conv beat the residual champion — is the remaining (compute-bound) step.

**Channel schedule (`--conv-sched`): the V1-shaped inverted funnel does not beat
constant width on digits (negative, preliminary).** This is where the width-
schedule idea (task 16: budget = *average* channels, vary the shape) meets conv
(task 28), tested in one knob. Motivation from biology: the primate visual
pathway is roughly an inverted funnel — retinal ganglion cells ≈ LGN (~1:1),
then LGN → V1 fans out 17–40× ([PMC5750718](https://pmc.ncbi.nlm.nih.gov/articles/PMC5750718/)) — so "wide first layer" seemed principled.
At matched average channels (~61, residual, tree3):

| shape | schedule | digits acc |
|---|---|---|
| constant | 64,64,64,64,64 | **0.9044** |
| inverted funnel (V1-like) | 128,64,48,32,32 | 0.8867 |
| funnel | 32,32,48,64,128 | 0.8111 |

Constant width wins; the inverted funnel is slightly behind and the funnel is
far worse. Two honest reads: (1) the *direction* is right — starving the
first layer (funnel) that sees the raw image costs 9 pt, so "capacity early"
matters, just not enough to beat constant here; (2) digits (8×8) is a poor
testbed — pooling collapses the map to 1–2 px by layer 3, leaving no room for
a spatial schedule, and the wide first layers thrash/OOM on 6 GB. A fair test
needs MNIST (28×28) plus the memory work; deferred. The `--conv-sched`
mechanism itself does exactly what task 16 asked (per-layer width at a fixed
*average* budget), now on top of convolution.
