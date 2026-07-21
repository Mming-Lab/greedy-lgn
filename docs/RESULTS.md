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
greedy to 90.9% and the method headline (residual+skip) to **94.6%** (3-seed mean 94.64),
while e2e is stuck at 81.8% because it cannot go deep. Net: on the same single-500 budget,
**backprop-free greedy beats end-to-end backprop by +12.8 pt on MNIST** (94.6 vs 81.8),
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
| residual + `--skip-input` (cap 40, `--patience 2`) | 93.82% (seed 1; 3-seed mean 93.72%) | 34 |
| residual + `--skip-input` (`--patience 10`) | **94.36%** (seed 1; 3-seed mean **94.64%** = headline) | 41 (seed 2 goes to 95) |

digits likewise: residual 96.4% → residual+skip **97.3%**. The headline is the 3-seed mean **94.64%** (verified; see the depth-exploration section below for the per-seed numbers) — a single 500-gate net with no ensemble and no width scaling, it beats the previous repo best of 90.9% (which needed 4,000-gate layers × 4 ensemble ≈ 16× the gates plus voting). Honest cost: skip's +3 pt comes with a much deeper circuit (9 → 41–95 layers = longer critical-path latency), so residual-alone (90.86% @9) is the shallow/fast champion and residual+skip (94.64%) is the max-accuracy champion — pick by whether latency or accuracy matters.

**Residual also revives the carry window (①+②).** Windowed lookahead was near-useless on plain greedy, and the carry-forward window (`--carry`, keeping the uncommitted lookahead layer instead of discarding it) actually *hurt* on its own (digits 90.2 vs 91.6). On top of residual, both flip positive: `--window 2 --commit 1 --win-loss all` adds +1.1 pt (MNIST 90.86 → 92.00), and adding `--carry` adds a further +0.25 pt (→ 92.25). The sign flip is the point — a carried layer's value in plain mode is its (fragile) features, which go stale when the layer below is frozen; in residual mode its value is the *answer correction* it contributes, which stays portable. So the project owner's "①+② should revive ②" was right in direction, though the gain is small and still below residual+skip (94.64%; 93.72% under the `--patience 2` rule these runs used). Residual now supports `--window`; W=1 stays bit-exact.

Full run log: [issue #11](https://github.com/Mming-Lab/greedy-lgn/issues/11).

## Depth exploration (`--patience 10 --max-layers 80`): the champion was being stopped too early

The depth-selection rule stops adding layers after `--patience` consecutive
non-improving layers (default 2) and rolls back to the best depth. Because a
frozen layer is never retrained, the depth-*d* network is always a **prefix**
of the final one — `run_greedy` already returns `layers[:best_depth]`, so
searching deeper costs time and nothing else (the rollback is free). That made
`--patience 2` worth questioning: the champion's 3 seeds all stopped at depth
26–34, i.e. on patience, not at the depth cap.

digits agreed first (3 seeds forced to depth 60, then re-scored under each rule:
p=2 mean 96.96 → p=10 mean 97.85, +0.89 pt, 3/3 seeds). MNIST is the referee:

| MNIST, residual+skip, 500 gates | `--patience 2` (old) | `--patience 10` | Δ | stopped at |
|---|---|---|---|---|
| seed 1 | 93.82% @34 | **94.36% @41** | +0.54 | 51 (patience) |
| seed 2 | 93.48% @26 | **95.24% @95** | +1.76 | 105 (patience) |
| seed 3 | 93.87% @29 | **94.32% @43** | +0.45 | 53 (patience) |
| **mean** | **93.72%** | **94.64%** | **+0.92 (3/3 seeds)** | |

**94.64% is the new method headline** (single 500-gate net, no ensemble, no width
scaling). All three seeds stop on *patience*, not on the depth cap, so the
protocol is uniform across them. Every run verified bit-exact
(`identical = True`). Cost: 7.6–11.8 h per seed on a 6 GB RTX 3060 (seed 2 took
27,406 + 4,082 s); this only completed because `--checkpoint` let each run be
split across sessions. Note how wide the depth spread is — seed 2 stays productive
to depth 95 while seeds 1 and 3 are done by ~42.

(An intermediate **94.53%** was reported when the batch first ran with
`--max-layers 80`: seed 2 hit that cap at 94.91% @78 while still climbing, so the
mean mixed a cap-truncated seed with two patience-stopped ones. Resuming seed 2
alone from its checkpoint with `--max-layers 120` closed that gap — 95.24% @95,
stopping on patience at 105 — and made the protocol uniform. 94.53% is retired.)

**Honest caveats.** (1) The probe is the *test* set, so searching deeper also
draws more samples from a noisy criterion — `patience 10` picks the best of ~10
extra depths where `patience 2` picked the best of ~2, and seed 2 drew from 105.
The old headline was selected the same way so the comparison's direction holds,
but the peak values are likely ~0.1–0.2 pt optimistic (seed 2's depths 90–105 sit
in a 94.99–95.24% band; the reported peak is the top of it). A clean fix needs a
validation split — a protocol change, not attempted here. (2) Each seed's stopping
depth is now its own, so the headline is not a fixed-depth claim.

Full run log: [issue #14](https://github.com/Mming-Lab/greedy-lgn/issues/14). The three
frozen circuits are published as release assets
([`task29-checkpoints`](https://github.com/Mming-Lab/greedy-lgn/releases/tag/task29-checkpoints)) —
rebuild them with `--checkpoint` or `tools/vote_checkpoints.py` to re-verify these
numbers, or to extend seed 2 past the cap, without repeating the 8–11 h per seed.

## Depth curve: it fits a saturating law, but that does not buy cheap early stopping (negative)

With three 51–80 layer curves in hand, the obvious question was whether depth ×
accuracy follows a law you could *extrapolate* — fit a few layers, predict the
rest, stop early. The fit itself is excellent (a plain log curve is not: R² ≈
0.76–0.81):

| MNIST, residual+skip | Hill `a − b/(1+(x/k)ⁿ)` | power `a − b·x⁻ᶜ` |
|---|---|---|
| seed 1 (51 layers) | R² **0.9988**, asymptote 95.17% | R² 0.9975, 96.17% |
| seed 2 (80 layers) | R² **0.9990**, asymptote 95.36% | R² 0.9972, 96.19% |
| seed 3 (53 layers) | R² **0.9976**, asymptote 95.57% | R² 0.9940, 97.79% |

The Hill form fits best in-sample, and at this point its asymptote looked like the
trustworthy one — consistent across seeds (95.2–95.6%) where the power law's was
not (96.2 / 96.2 / **97.8**). **That reading was wrong, and extending seed 2 proved
it.** Resuming seed 2 to depth 105 and scoring the 80-layer fits against the new
ground truth:

| depth | actual | power (fit on 80) | Hill (fit on 80) |
|---|---|---|---|
| 90 | 95.15% | 94.93% | 94.72% |
| 95 | **95.24%** | 94.98% | 94.75% |
| 100 | 95.20% | 95.03% | 94.78% |
| 105 | 95.13% | 95.07% | 94.81% |

Against the depth-90–105 band (mean 95.11%), the power fit is off by **−0.11 pt**
and Hill by **−0.35 pt** — *the better in-sample fit extrapolates worse*, and both
now **under**-shoot where the early fits over-shot. Worse for Hill: its 80-layer
asymptote said 95.36% was the ceiling at infinite depth, and the run reached 95.24%
by depth 95 — refitting on 105 layers moves that "ceiling" up to 95.70%. The
apparent cross-seed consistency was systematic under-estimation, not stability. The
power law's asymptote is the steadier one (96.68 → 96.32 → 96.19 → 96.26% as layers
40 → 60 → 80 → 105 arrive), but "steadier" is not "verified": no fit here has been
tested beyond ~1.3× the fitted range.

**But the useful version of the idea fails.** Fitting only the first N layers and
predicting the final layer's accuracy (Hill, error in pt):

| fit on | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| N=15 | +0.38 | −0.30 | **+1.89** |
| N=20 | +0.44 | −0.32 | **+1.87** |
| N=30 | −0.09 | −0.50 | +0.77 |
| N=40 | −0.09 | −0.35 | +0.46 |

On two seeds a 15-layer fit lands within ±0.4 pt, which looked like a way to
cut experiment time by ~4×. **seed 3 breaks it** (+1.9 pt at N=15, still +0.5 pt
at N=40) — larger than most of the levers in this document. Extrapolated
asymptotes are therefore not a usable stopping oracle at 3 seeds, and the
"calibrate on 20 layers, then decide" protocol is **not adopted**. (A cheaper
honest reading: the early fit can't yet see how hard the curve bends; a shift
parameter fixes the *average* case but not the variance across seeds. Note the
error is not even one-signed — early fits over-shoot, the 80-layer fits
under-shoot.) The curve fit stays a *descriptive* observation, and the
model-selection lesson generalizes past this repo: **in-sample R² ranked the two
candidates in exactly the wrong order for extrapolation**, and only running the
experiment further could tell.

Related and worth stating: **the residual readout alone does not produce this
curve — residual+skip does.** Forced to depth 60 on digits, residual-without-skip
peaks at 96.4% @10 and then **decays to 92.0%**, so no saturating fit applies
(R² collapses to 0.48–0.51); residual+skip climbs monotonically and fits at
R² ≈ 0.98–0.99 (also at half width, 250 gates). Depth-tolerance is a property of
the *combination*, consistent with the two levers curing different failure modes
(the earlier issue-#11 reading of "residual holds ~96%" came from a 16-layer cap).

## Numerical footnote: why a CUDA run and a CPU run report different accuracy

The residual probe on GPU and the CPU-side `simplify` check disagree slightly on
the same frozen circuit (e.g. seed 1 depth 41: probe **0.9436**, simplify
**0.9432**). This was previously logged as a vague "τ-division float tie"; the
mechanism is now pinned down, and it is **not** nondeterminism and **not** the
circuit:

- Gate selection (`logits.argmax`) has **zero exact ties** across all 41 layers ×
  500 gates, so CPU and GPU discretize to the *same* circuit. Per-layer class
  counts are exact integers (≤46, well inside float32's exact-integer range).
- The divergence is `group_sum`'s `counts / τ`. PyTorch's CUDA path is
  bit-compatible with **`counts × (1/τ)`** (reciprocal multiply) while the CPU
  path does a true divide. They differ by one ULP. Replaying seed 1's checkpoint
  on CPU reproduces the CUDA log at **51/51 depths** with `× (1/τ)` and only
  **10/51** with `/ τ`.
- The residual readout then *accumulates* that over 41–80 layers, so mathematically
  tied classes end up ordered by rounding noise: sample 195's classes 1 and 5 both
  total exactly 1070 bits, but sum to 151.3208618164 vs 151.3208770752.
- Impact: 4 samples at depth 41 (±0.04 pt), up to 17 samples (0.17 pt) at some
  depths. Only genuinely ambiguous samples move.

Consequences: bit-exactness claims are unaffected (they compare the circuit's
output bits, not the τ-scaled readout); headline numbers are all CUDA runs, so
they are compared like-for-like; and `tools/vote_checkpoints.py` sums **exact
integer counts** and is immune — the same fix issue #8 forced on ensemble voting.
Caveat this exposes: `tests.py` pins are device-dependent for residual configs
(the `conv C64/tree3 + residual` case reproduces 0.6422 only on the device it was
pinned on, and gives 0.64 otherwise) — its "GPU numbers are identical to CPU"
note does not hold for residual readouts.

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
reweight only doubles down. Does not touch the residual+skip champion (94.64%, 3-seed mean).

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

| config | ffres | + `--group-boost 3` | + `--group-review` | CE-residual champion |
|---|---|---|---|---|
| residual only | 95.41% | 95.92% (+0.51) | 95.63% (+0.22) | **96.4%** |
| residual + skip | 96.29% | 95.93% (−0.36) | 96.07% (−0.22) | **97.3%** |

**Honest verdict: loses to CE on both configs, consistently across all 3 seeds**
(not a 1-seed fluke — every seed underperforms the CE headline), and every attempt
to rescue it lands in the same 95.4–96.3% band with signs that flip between
configs, i.e. noise.

**A hypothesis this repo tested and did not confirm.** The natural explanation for
the ~1 pt gap was that CE's softmax bakes in *implicit hard-negative mining* —
gradient on the wrong classes concentrates on whichever one is currently most
confusable, weighted by its softmax probability — while the per-class independent
loss spreads it evenly over all 9. That predicts an explicit negative-focusing
mechanism should recover the gap, and there was a strong precedent: for the FF
objective, `--ff-neg review` (a misclassified sample studies *its own* wrong
answer) was the single biggest lever in this whole document, worth +3.86 pt.
`--group-review` ports exactly that rule to ffres — push down one wrong class per
sample (the frozen prefix's own mistake when it is wrong, a random wrong class when
it is right) instead of all 9 equally. **It does not close the gap** (+0.22 /
−0.22, sign-flipping like boost). So "ffres lacks CE's hard-negative mining" is now
evidence-against, not merely untested.

**Why ffres loses is therefore unresolved.** The remaining suspect is the
absolute-calibration cost — ffres has to place the accumulated score on the right
side of a threshold, work `argmax` never needed — but that is a guess, not a
result. Implementation caveat on the review test: it grades with the frozen prefix
accumulator, and FF's own findings say prefix-grading is the *weak* variant while
warm-up + self-grading is what made review work. The residual case is not obviously
the same trap (the accumulator is the current frozen network's actual prediction,
~94% accurate by depth 40, not a stale several-layers-back one), but a
warm-up/self-graded version has not been tried.

Checkpoint resume is verified bit-exact for this objective (a `groupsum` flag
combination with no new persisted state), and `--group-review` off reproduces plain
ffres bit-for-bit — its negative draw uses an independent generator, so the wiring
/ logits / minibatch RNG streams are untouched. MNIST untested — per this repo's
digits/MNIST protocol a MNIST reversal isn't ruled out, but a ~1 pt loss that
neither boost nor review rescued lowered this lever's priority rather than
justifying the GPU time.

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

**Revised below**: "retires the lookahead window" was measured on *plain*
greedy. On the residual readout the window is *not* retired — it helps, once
its discarded layers are carried (`--carry`). And `residual + warm-start` being
flat holds for accuracy but hides a gate-count effect. See the next section.

## Carrying the lookahead window (`--carry`) on the residual readout: stop throwing the layers away

`--window W --commit J` trains W layers jointly but freezes only the first J and
**discards the remaining W−J** — every slide rebuilds them from scratch
(receding horizon, `src/greedy.py`). `--carry` keeps them instead, handing the
trained weights to the next slide as scaffolding. Carry was measured before and
lost (digits 90.2 vs 91.6) — but that was on *plain* greedy, where what it
carried was brittle features. The open hypothesis (task 20) was that on the
residual readout it carries something else: a correction to a running answer.

digits, 3 seeds, 500 gates/layer, `--group-residual`, `--window 4 --commit 1`.
All 12 runs bit-exact (`identical = True`):

| config | mean acc | seeds | gates after simplify |
|---|---|---|---|
| residual only (`--window 1`) | 96.30 | 96.00 / 97.11 / 95.78 | 2126 |
| + window 4, layers discarded | 96.81 | 97.33 / 96.00 / 97.11 | 2790 |
| **+ window 4, `--carry`** | **97.41** | 97.11 / 97.33 / 97.78 | 2861 |
| + `--carry --warm-start 5` | 97.48 | 98.00 / 97.33 / 97.11 | **1600** |

Two results, and they are different in kind.

**The window works on the residual readout, and carrying pays.** Residual +
carried window beats residual alone by **+1.11 pt** and the discarding window by
**+0.60 pt** (2 of 3 seeds). Carry is also the steadier of the two (spread 0.67
vs 1.33) — consistent with the mechanism, since the discarding variant redraws
three layers from a fresh RNG on every slide. Note the residual-only baseline
reproduces the 96.30 recorded in the warm-start section above, on a separate
run, which is the cross-check that these numbers are comparable.

**Warm-start is flat on accuracy (+0.07 pt, i.e. nothing) but shifts the
accuracy-per-gate curve.** Same raw circuit (~3.8k gates before simplify), but
60% of it is removable versus ~22% for the others: identity init leaves many
gates as passthroughs and the simplifier deletes them. Squeezing the width shows
this is *not* free, and the first framing ("−44% gates at no cost") was
over-provisioning on digits at 500 gates:

| `--gates` | carry | carry + warm-start 5 | Δacc | Δgates |
|---|---|---|---|---|
| 500 | 97.41 / 2861g | 97.48 / 1600g | +0.07 | −44% |
| 200 | 94.81 / 1206g | 94.44 / 792g | **−0.37** | −34% |
| 100 | 89.33 / 624g | 88.52 / 461g | **−0.81** | −26% |

The tighter the budget, the more accuracy warm-start gives up and the less it
saves. But compared at *matched gate count* rather than matched `--gates`, it
still wins: 97.4% costs 2861 gates with carry and 1600 with warm-start (−44%);
94.8% costs 1206 versus 792 (−34%). The per-`--gates` loss is the price paid for
gates removed, and the exchange rate is favourable.

Caveats: digits only, and digits at 500 gates is over-provisioned — the MNIST
conv runs were all capacity-limited (train ≈ test), the regime where deleting
60% of a circuit is most likely to cost accuracy. MNIST is the judge and has not
ruled. Idea proposed by the project owner ("stop discarding them") after
noticing the window was unused in the headline.

### `--carry` works with skip after all

`--carry` was gated to no-skip ("carried layers assume constant `in_dim`"). That
restriction was wrong — it generalised from the no-skip case. Carrying moves a
layer from window position `J+i` to position `i`, and in every skip mode both
the width and the *content* of what it reads are preserved:

- **skip-input**: the pool is `[X ∥ h]` with `h` the previous layer's bits, so
  for `d0 ≥ 1` every window position has `in_dim = |X| + gates`. The carried
  layer read `[X ∥ win[0] output]` at position 1; at position 0 of the next
  slide it reads `[X ∥ last committed layer output]` — the same layer.
- **skip-all**: position `k` has `in_dim = pool(d0) + k·gates`, and committing
  `J` layers grows the pool by exactly `J·gates`, so `J+i → i` lines up. The
  appended bits are the committed layers, which are exactly the window layers
  that preceded the carried one.

Lifting the gate (digits, 3 seeds, 500 gates, `--skip-input --window 4
--commit 1`, residual):

| config | mean acc | seeds |
|---|---|---|
| skip + window 4, layers discarded | 97.04 | 97.11 / 96.22 / 97.78 |
| **skip + window 4, `--carry`** | **97.56** | 97.78 / 96.89 / 98.00 |

**+0.52 pt, 3 of 3 seeds** — carry pays with skip as well as without (+0.60 pt
no-skip), and the combination is the best digits config measured here. `--carry`
is also no longer exclusive with `--checkpoint`: the uncommitted lookahead
layers are persisted too, so a resumed run is bit-identical to an uninterrupted
one (`tests.py` pins both, no-skip and skip-input). Restriction spotted by the
project owner ("skip should work at the same time, I'd think").

### MNIST verdict: carry replicates, and buys a smaller circuit

MNIST, seed 1, matched protocol (`--batch 512 --group-residual --skip-input
--max-layers 120 --patience 20`), the only difference being the window and
carry. Both runs bit-exact after simplification (`identical = True`):

| config | acc | depth | gates before | after simplify | wall clock |
|---|---|---|---|---|---|
| window 1 (baseline) | 95.28 | 111 | 55,500 | 43,396 | 1.5 h |
| **window 4 + `--carry`** | **95.76** | **93** | **46,500** | **37,123** | 5.7 h |

**+0.48 pt**, and it wins on every circuit measure at once: 18 fewer layers and
**14.5% fewer gates** after simplification. The digits result (+0.52 pt, 3
seeds) replicates on MNIST in both sign and size.

The honest costs: **3.9× the training time** for that +0.48 pt — carry wins on
gate efficiency and loses on compute efficiency — and this is **one seed** on
MNIST. The window and carry are also both switched on at once here, so their
individual contributions are not separated (on digits the window helped alone
and carry added on top).

### A batch/depth confound, chased down and dismissed

The batch-512 baseline above (95.28, depth 111) sits well above the published
94.64% (3-seed mean, depths 41/95/43, full batch). That gap looked at first like
a batch effect — I wrongly attributed it to `--batch 512` twice before actually
isolating it. It is almost entirely the **depth search**, not the batch.

The published 94.64% used `--patience 10`, which stopped two of three seeds
early (depths 41 and 43). Extending those same full-batch seeds deeper (the
depth-curve runs) reached 95.01 / 95.24 / 95.39 at depths 79 / 95 / 117 — a
3-seed mean of **95.21%**, still full batch. Against the batch-512 deep runs
(95.28 / 95.24 / 95.66, mean **95.39%**), the difference is **+0.18 pt** (+0.27 /
0 / +0.27), within seed noise:

| 500-gate, residual+skip, 3 seeds | mean | depths |
|---|---|---|
| full batch, `--patience 10` (published headline) | 94.64 | 41/95/43 |
| full batch, deep search | 95.21 | 79/95/117 |
| batch 512, deep search | 95.39 | 111/90/109 |

So batch size is a **speed** lever, not an accuracy one (full batch materializes
a [rows × gates × 16] tensor that thrashes a 6 GB GPU; batch 512 is ~15× faster
per layer at the same accuracy). The published 94.64% was low because the depth
search was cut short on two seeds, not because of the batch setting. The
headline number is not revised on the strength of this — a deeper `--patience`
buys ~+0.5 pt but also increases the test-set bias of choosing depth on the
probe, the same caveat noted in the depth-exploration section.

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
low plane on the champion (residual + `--skip-input`, cap 40, `--patience 2`),
each seed run side-by-side with a same-protocol control. Note both columns predate
the depth-exploration result above, so the control here is the *old* 93.72%
headline; whether the low plane still adds +0.55 pt on top of the deeper
`--patience 10` champion (94.64%) has **not** been measured:

| residual + skip, MNIST | control (63,127,191) | + low plane (31,63,127,191) |
|---|---|---|
| seed 1 | 93.82% @34 | 94.08% @27 |
| seed 2 | 93.48% @26 | **94.60% @40** (still climbing at the cap) |
| seed 3 | 93.87% @29 | 94.13% @33 |
| **mean** | **93.72%** | **94.27%** (+0.55, **3/3 seeds**: +0.26/+1.12/+0.26) |
| bit-exact | identical = True (all) | identical = True (all) |

**Against its own control the low plane adds +0.55 pt (93.72% → 94.27%, best
single 94.60%).** But it changes the *input encoding* (a fourth threshold plane =
more input bits), not the network or the learning, so it is credited **off the
arena** — not as the headline. (The method headline has since moved to 94.64%
via depth exploration, which this comparison predates — see above.) Every run is verified
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

**MNIST referee (2026-07-18): conv loses to the dense residual champion, and
the reason is a hard depth ceiling, not width.** The overnight batch (single
network, residual readout, `--patience 10`, batch 2048, seed 1, `--epochs 120`)
pitted conv against the 500-gate dense method (dense residual+skip, 3-seed
**94.64%**):

| config | best test | depth | stop | vs banner |
|---|---|---|---|---|
| conv C32/tree3 | 83.52% | 9 | patience | −11.1 pt |
| conv C64/tree3 | **89.49%** | 9 | patience | −5.15 pt |
| conv inverted-funnel 128,64,32 | — | — | OOM at layer 1 | — |
| conv C32/tree3, `--epochs 240` | 84.68% | 8 | budget-cut¹ | −9.96 pt |
| dense residual+skip (banner) | **94.64%** | 41–95 | — | — |

¹ stopped by the batch's time cap via `--stop-file`, not a clean patience stop
(but it had already peaked at layer 8 and was declining, so it is near its
ceiling regardless).

Three diagnostics, all pointing the same way:

1. **Conv's usable depth runs out at ~9 layers — this is the real limiter.**
   Both C32 and C64 peak at *depth 9*. Doubling the channels lifts the ceiling
   (83.5→89.5%) but does **not** buy more depth. The cause is geometric:
   `--conv-pool 2` collapses the feature map 28→14→7→3→1 in four layers, and
   after that each layer only carries the C-bit channel vector (32/64 bits),
   crawling as a narrow dense residual net. The banner's dense net stacks depth
   41–95 to reach 94.64% — a different *shape* of resource. **Width buys the
   ceiling; it cannot buy depth.**

2. **Every config is capacity-bound, not overfit.** train ≈ test throughout
   (C32 tops out at train 0.82, C64 at train ~0.90 with test tracking within
   ~1 pt until the very end). The circuits cannot even fit the training set —
   this is not a generalization failure.

3. **Conv on MNIST saturates by 120 epochs; the digits "240" is not
   extrapolable.** Layer-by-layer, `--epochs 120` and `240` differ within
   noise (240 peaks at 84.68% vs 120's 83.52%, well inside seed-level scatter).
   The earlier digits sweep (conv saturates ~240 ep, twice dense) does **not**
   carry to MNIST — the "keep conv ≥120" caution can be relaxed to "120 is
   enough" here.

Two follow-ups the verdict sets up (both need a small code change, not yet
done): (a) the inverted funnel is still unmeasured — C128's first layer OOMs in
`commit()`, where `hard_chunked` materializes a full-resolution float feature
map (60000×128×14×14 ≈ 6 GB) before the uint8 cast; returning uint8 per chunk
fits it. (b) The real lever suggested by diagnostic 1 is a **pooling schedule**:
pool only in the first layer or two (stop at 7×7), then `pool=1` to spend the
remaining budget on *depth* instead of collapsing the map. 7×7 layers are cheap
(~10 min each on this GPU), so 20 of them fit in ~3 h. `--conv-pool` is
currently uniform across layers; making it per-layer is task 28's next step.

**Pooling schedule (`--conv-pool-sched`, 2026-07-18): the depth-lifetime
diagnosis is confirmed — keeping the map at 7×7 breaks the flat ceiling — but the
limiter then shifts from depth to channel capacity.** Implemented follow-up (b):
`--conv-pool-sched 2,2,1` pools in layers 1–2 (28→14→7) then holds 7×7 (`pool=1`)
so later layers stay real conv layers instead of collapsing to 1×1. C64, same
protocol (residual, `--patience 10`, batch 2048, seed 1, `--epochs 120`):

| C64 conv | best test | depth | behaviour |
|---|---|---|---|
| flat `--conv-pool 2` (28→14→7→3→1) | 89.49% | 9 | collapses to 1×1 by layer 4, plateaus |
| **pool-sched 2,2,1 (holds 7×7)** | **90.52%** | 8 | +1.03 pt, first to cross 90% |

Layers 1–2 are bit-identical to the flat run (same pool=2, wiring, seed); the
paths diverge at layer 3, where holding 7×7 gives +2 pt (80.75 vs 78.77) and by
layer 6 the schedule crosses 90% — reaching flat's *entire-run best* at shallower
depth. So the flat plateau really was the 1×1 collapse, and the mechanism
(receptive field must stay alive to keep stacking productive depth) holds.

But the win is modest (+1 pt), because past depth 8 the schedule runs into a
**different** wall: C64's channel capacity. The overfitting signature is textbook
— depth 8 train 0.908/test 0.905 (matched), depth 10 train 0.913/test 0.905
(train pulls ahead), depth 14 gap 2.1 pt, depth 16 train itself declines
(residual boosting now hurts). Unlike the flat runs (capacity-bound, train≈test),
the pool-sched run *does* overfit — it has enough usable depth to start
memorizing, so the bottleneck moved from "not enough depth" (fixed) to "not
enough channels". The natural next step is C128 on the same schedule, which the
memory fix below now unblocks.

**Memory fix (`hard_chunked` → uint8, 2026-07-18): C128 commit no longer OOMs.**
The follow-up (a) fix: `hard_chunked` now casts each batch-chunk to uint8 before
concatenating (hard bits are exactly 0.0/1.0, so the cast is lossless), and a new
`group_sum_hard` reduces the GroupSum over sample-chunks (group_sum is per-sample,
so chunking dim 0 is bit-exact). Together they keep the full-resolution feature
map from ever existing as float — the 6 GB C128 map is 1.5 GB as uint8. All 13
regression pins stay bit-identical (the digits conv pins run one chunk =
unchanged; the fix only changes peak memory on configs that previously OOM'd),
and C128's layer-1 commit — the exact point that OOM'd before — now passes.

## Width unlocked (8,000 gates/layer): the memorization wall

*(Migrated from the retired `WHITEPAPER2.md`, 2026-07-21, with the accuracy and
gate figures corrected against the run logs — see the note at the end.)*

Lifting the per-layer budget to 8,000 gates (`--batch 1024 --epochs 60 --gates
8000 --group-residual --skip-input`, window 1, so no gradient crosses a layer)
reaches **97.11%** as a 3-seed mean. Then it stops, hard. Seed 1's depth sweep:

| depth | train | test | |
|---|---|---|---|
| 8 | 99.86% | 96.90% | |
| **10** | **100.00%** | **97.00%** | training set fully memorized |
| 14 | 100.00% | 96.95% | |
| 18 | 100.00% | 97.04% | +8 layers bought **+0.04 pt** |

At depth 10 the network has memorized all 60,000 training images. Eight more
layers — 64,000 more gates, 24 more minutes — bought four test samples. Not
*zero*, curiously: test creeps 97.00 → 97.04 with train pinned at 100%, which is
textbook boosting behaviour (margins keep widening after training error hits
zero) and the residual readout *is* boosting. At 8,000 gates a layer it is not a
trade worth making.

Width scaling says the same thing from the other side:

| width | best test | raw gates at best |
|---|---|---|
| 500 | 94.64% | 39.5k–58.5k |
| 2,000 | 96.61% | 80,000 |
| 8,000 | 97.11% | 72k–144k |

Each 4× in width buys less: error shrinks ~0.85× per doubling, far from the
~0.7× the early 500→2,000 scaling suggested. Extrapolating that law predicted
**97.65%** at 8,000 gates; the measured 97.11% is **0.5 pt short**. There is a
floor around 2–3% error that width alone is not touching. **Past a point, width
buys memorization, not generalization.**

### Gate efficiency: this loses clearly

Absolute accuracy is one thing; **gates per point of accuracy** inverts the
picture:

| model | training | gates | MNIST |
|---|---|---|---|
| [LILogic Net](https://arxiv.org/abs/2511.12340) (2025) | gradient-based, learnable connectivity | **8,000** | **98.45%** |
| [difflogic](https://arxiv.org/abs/2210.08277) small (2022) | e2e backprop | 48,000 | 97.69% |
| difflogic large | e2e backprop | 384,000 | 98.47% |
| this repo, 3-seed mean | layer-local only (window 1) | ~86,000 | 97.11% |
| this repo, 2,000 × depth 40 | layer-local only (window 1) | ~65,000 | 96.61% |

LILogic Net reaches **+1.3 pt on ~10× fewer gates**. That is not a near miss; it
is a different league, and it comes from optimizing the whole network at once
(there, including the wiring itself) — exactly what layer-local training gives
up.

Reshaping the same budget does not recover it:

| shape at ~18k gates | test | | shape at ~48k gates | test |
|---|---|---|---|---|
| 500 × 36 | 93.6% | | 500 × 96 | 95.2% |
| 2,000 × 9 | 94.5% | | 2,000 × 24 | 96.1% |
| 8,000 × 2 | 92.3% | | 8,000 × 6 | 96.6% |

Aspect ratio matters (too thin wastes gates on redundancy; too fat starves the
boosting chain of correction steps), but no shape tried here closes the gap.
Hypothesis, not a measurement: each layer optimizes for *itself*, so it hoards
information later layers duplicate — the 500-gate circuit diagnostics measured ~50%
functional redundancy, which fits.

### What 6 GB actually constrained

- 8,000 gates: comfortable (batch 1024, ~2.5 GB, 3 min/layer)
- **16,000 gates: OOM — but not in training.** Training fits. The pool
  *transition* did not: the code built the next layer's wiring pool with
  `cat([X, h])`, briefly holding the old pool, the new pool, and the layer's
  output at once (~3.2 GB of transient at that width). Fixed by allocating the
  pool once and writing the output in place — under `--skip-input` the input
  half never changes. Same bytes, bit-exact (pinned by the regression suite).
  16,000 gates then trains fine, just slowly on this GPU.

**The wall at 16k was the implementation, not the method or the hardware.** The
8,000-gate runs were never memory-bound.

### Correction to the previously published figures (2026-07-21)

The retired whitepaper and README quoted **"97.04%, ~66k gates, 3 seeds"**. That
pairing was a mix-up: the run logs give three seeds of 97.04 (depth 18, 118,699
gates), 97.15 (depth 9, 59,808) and 97.13 (depth 12, 79,182). So:

- **3-seed mean at each seed's best depth: 97.11%, ~86k gates** — the honest
  representative figure
- 97.04% / 118.7k gates = seed 1 alone
- 97.00% / ~66k gates = seed 1 truncated to depth 10

The published "97.04% / ~66k" took the accuracy from one depth and the gate
count from another. The corrected accuracy is *higher* than published; the
corrected gate count is also higher.

## What the circuits actually look like (diagnostics)

*(Migrated from the retired `WHITEPAPER1.md`, 2026-07-21.)*

Two standalone tools ([`tools/diagnose.py`](../tools/diagnose.py),
[`tools/dynamics.py`](../tools/dynamics.py)) inspect the trained circuits:

- **Warm-start has a fingerprint.** Plain greedy spreads over the 8 simple gates
  (~9–11% each, constants ~0.5%); warm-start collapses the distribution onto
  gate A (pass-through, 39%; A-family ~70%). The identity bias is literally
  visible in the learned circuit, and it explains why warm layers are barely
  touched by training (62% keep their init) and simplify less (their
  pass-throughs are live).
- **~50% functional redundancy.** Half the gates produce an output column
  identical or complementary to an earlier gate — yet the simplifier's
  *structural* duplicate-merge finds zero (different wiring, same behaviour on
  the data). A quantified gap, honestly not the same as logical equivalence.
- **Recurrent cells are fading-memory maps, with real oscillators.** Rolling a
  trained `--seq` cell forward under blank input: the accuracy winner
  (warm-start) is a contractive map converging to a single fixed point in 2–4
  steps — an echo-state / fading-memory regime. But plain `--seq` spontaneously
  grows period-2 and period-6 limit cycles (oscillators built from logic gates
  and a register), just on the losing configuration.

## What this borrows

Almost every ingredient is prior work; this section is about being explicit, not
claiming credit.

| piece | where it comes from |
|---|---|
| Logic gate networks, FPGA-friendly inference | [difflogic](https://github.com/Felix-Petersen/difflogic) (Petersen et al., NeurIPS 2022) — the platform, not mine |
| Convolutional logic gates, post-training synthesis | [conv-DLGN](https://arxiv.org/abs/2411.04732) (NeurIPS 2024) |
| Recurrent logic gates, temporal state | [RDDLGN](https://arxiv.org/abs/2508.06097) — `--seq` follows its wiring |
| Residual readout | plain boosting / deep supervision (standard) |
| Grow-and-freeze, adaptive depth | Cascade-Correlation (Fahlman & Lebiere, 1990) |
| Local per-layer objectives | Forward-Forward (Hinton, 2022); block-wise greedy (Belilovsky et al., 2019) |

I have not surveyed the literature properly. If any of this — or the combination
— duplicates prior work, that is expected; please open an issue so I can point
to it.

## How much backprop is actually used (2026-07-21)

Worth stating precisely, because the repo's older framing overclaimed.

**With `--window 1` (the default, and what every published number here used):
no gradient crosses a layer.** Each layer is trained alone against a local loss,
then discretized and frozen.

**With `--window W` for W > 1, gradients cross W layers.** The window's layers
are chained in one autograd graph (`next_pool` concatenates the previous layer's
output with no `detach`), the optimizer holds every window layer's parameters,
and with the default `--win-loss last` the loss on the final window layer
backpropagates through all W. That is backpropagation, bounded to W layers.

The distinction that survives: gradients never enter the **frozen** prefix, so
the distance a gradient travels is capped at W regardless of total depth (93+
layers in the deepest runs here). That is still structurally different from
end-to-end training, but "backpropagation-free" is only accurate at W = 1.

Results obtained with a window must be labelled as such. In this document that
means the lookahead-window rows of the fixed-budget ladder (W = 2) and the
`--carry` results (W = 4).
