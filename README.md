# greedy-lgn

**Backpropagation-free, layer-by-layer training of Differentiable Logic Gate Networks** — with immediate discretization, adaptive depth, and incremental logic simplification.

> Train one logic layer at a time with a local loss, discretize it, freeze it, and let the next layer learn on *real* 0/1 bits. Stop adding layers when accuracy plateaus. Simplify the circuit as it grows.

Proof-of-concept. Runs on CPU in a few minutes. A single self-contained script, no dependencies beyond `torch` and `scikit-learn`.

📄 **Narrative write-ups** (reproducible experiment logs, each with a Japanese abstract):
> - **[Vol. 1](WHITEPAPER1.md)** — the climb to **94.64%** on a fixed budget (500 gates/layer), and what each idea taught.
> - **[Vol. 2](WHITEPAPER2.md)** — budget unlocked: **97.04%** on MNIST (3 seeds, ~66k gates, bit-exact, 40 min on a 6 GB laptop GPU), the memorization wall it hit, and the gate-efficiency fight it lost.

**Where this stands, in one table** (MNIST, single network, no gradient ever crosses a layer):

| | accuracy | gates | note |
|---|---|---|---|
| this repo, fixed budget (Vol. 1) | 94.64% | ~40k | 500/layer × depth 41–95, 3 seeds |
| **this repo, unlocked (Vol. 2)** | **97.04%** | **~66k** | 8,000/layer × depth 10, 3 seeds |
| [difflogic](https://arxiv.org/abs/2210.08277) (e2e backprop) | 97.69% | 48k | the platform this borrows from |
| [LILogic Net](https://arxiv.org/abs/2511.12340) (gradient-based) | 98.45% | 8k | far better gates-per-point |

Every number here is the **hard circuit's**, verified bit-exact against the trained network. On gate efficiency this repo loses clearly — the interesting part is the constraint (layer-local training only), not the number.

> **Just me playing around, not research.** I don't read papers — I bounce ideas off an AI assistant, run the experiments, and enjoy watching the accuracy points move. That's the whole thing. Nothing here is peer-reviewed, and the only "literature search" was asking the AI, so I make **no novelty or priority claims**. Plenty of these ideas probably already exist under names I don't know. Read it as a reproducible playground log, not a contribution; if it duplicates prior work, that's expected — pointers are welcome via an issue.

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

## Two kinds of number, and why

**500 gates/layer, one net, is the measuring stick — not the ceiling.** Ideas get compared at that fixed budget so that a lever's win is about the lever, not the spend (a bad idea dies on digits in seconds; MNIST settles it within the hour). Every table below is measured there. The accuracy headline comes from lifting the budget instead: **97.04%** ([Vol. 2](WHITEPAPER2.md)).

**Ensembles and input binarization sit outside both** — the first buys latency rather than accuracy-per-gate ([SCALING.md](docs/SCALING.md)), the second changes the data rather than the method.

Greedy (this repo, no cross-layer backprop) vs end-to-end backprop at the fixed budget, hard-circuit test accuracy on both datasets:

Hard-circuit test accuracy, 500 gates/layer, single net:

| dataset | plain greedy | greedy, best lever | end-to-end backprop |
|---|---|---|---|
| **digits** (8×8) | 88.2% | **96.4%** (residual) | 93.6% |
| **MNIST** (28×28) | 74.3% | **94.6%** (residual+skip) | 81.8% (peaks @depth 6) |

Why backprop can't be pushed further — and greedy can:

| property | greedy (this repo) | end-to-end backprop |
|---|---|---|
| discretization gap | **0 (by construction)** | present, grows with depth (+0.3→1.3 pt) |
| usable depth | 80+ (residual+skip productive to 41–95) | **collapses past ~6–8** (vanishing gradients) |
| float logits during training | **one layer** (500×16) | whole net (depth × 500×16) |

Two things to read off this. **At the plain starting line, greedy loses to backprop** (~5 pt digits, ~7 pt MNIST) — the honest cost of refusing cross-layer gradients. But **once the levers are on, greedy overtakes backprop, and the gap widens with dataset size**: +2.8 pt on digits, **+12.8 pt on MNIST** (94.6 vs 81.8). The reason is in the last two rows — e2e backprop can't use depth (gradients vanish past ~6 layers, so its single-500 net peaks at 81.8% @depth 6 and then collapses), whereas greedy has no cross-layer gradient to vanish and stays productive to 40–80 layers. From the plain starting line, the idea ladder (hard-circuit test accuracy, gap structurally zero throughout):

| idea (500 gates, single net) | digits (3 seeds) | MNIST |
|---|---|---|
| plain greedy (GroupSum + CE) | 88.4% | 74.3% |
| + skip-input (`--skip-input`) | 89.1% | —¹ |
| + windowed lookahead (`--window 2 --commit 2`) | 90.4% | 76.6% |
| Forward-Forward objective (`--objective ff`) | 86.0% | 76.8% |
| FF + window | 88.0% | 78.2% |
| FF + window + hard-negative mining (`--ff-neg`) | 89.7%² | 82.0%³ |
| **residual/boosting readout (`--group-residual`)** | **96.4%** | **90.9%** |
| **residual + skip-input** | **97.3%** | **94.5%⁴** |

¹ skip alone is a depth/width-synergy lever — modest at 500-gate single net, and it actively *hurts* FF, so it isn't a fixed-budget winner. ² `mix` negatives. ³ `review` + 0.5 warm-up. ⁴ **94.64% (3-seed mean, 94.36/95.24/94.32), depths 41/95/43** with `--patience 10` (all three stop on patience, not on the depth cap); the earlier 93.72% came from the default `--patience 2`, which stopped the same champion at depth 26–34 (+0.92 pt, 3/3 seeds — see [RESULTS.md](docs/RESULTS.md)). Residual-alone 90.9% converges at depth 9. Every run is **bit-exactly verified** through simplification (`identical=True`). This is the method headline — input binarization is held fixed at the default here (see the preprocessing note below for the +0.55 pt the input encoding buys on top of the *older* 93.72% champion; it has not been re-measured on the deeper one).

**The clear winner is the residual readout.** Plain greedy throws away every layer's class prediction except the last, which is exactly why accuracy decays with depth (shallow layers see fresh image info, but their good answers are discarded). Accumulate each layer's prediction instead — plain boosting — and the decay vanishes: MNIST climbs from 74.3% to **90.9%** (single 500-gate net, no skip/window/ensemble, not overfitting: train 91.9 / test 90.6). That already matches the scaling-track flagship below (4,000 gates × 4 ensemble) at a fraction of the area. Stacking `--skip-input` on top — residual accumulates the *answer*, skip re-exposes the *image*, different cures for the same decay — compounds to **94.6%** (3-seed mean 94.64, once the depth search is allowed to run to `--patience 10`), a new repo record, still at 500 gates single net (honest cost: depth grows 9 → 40–80, i.e. much more latency). The readout is plain boosting / deep supervision — the observation here is just that it fixes greedy-LGN's depth decay cleanly. [→](docs/RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes)

## Spending resources: width, depth, ensembles

Width (`--gates`) and depth buy accuracy by spending area, not by being clever — which is why they are kept out of the idea tables above. But "not a clever idea" isn't "not interesting": turning that one knob is what [Vol. 2](WHITEPAPER2.md) is about, and it took the same recipe from 94.64% to **97.04%** (8,000 gates/layer, depth 10, 3 seeds, bit-exact) — clearing the first line in 18 minutes, then hitting a memorization wall (train = 100%, test frozen at ~97.0%).

**Ensembles (`--ensemble`) are a different trade and stay off both arenas**: measured head-to-head, voting loses to depth at equal compute *and* equal gate count — 3 nets at depth 40 (94.86%) vs 1 net at depth 78 (94.91%). What it actually buys is **latency**: the same accuracy at half the critical path, for 1.5× the area. Details and the full budget sweep: **[SCALING.md](docs/SCALING.md)**.

## All experiments (details in [RESULTS.md](docs/RESULTS.md))

**Main track — fixed-budget ideas** (500 gates/layer, single net):

| experiment | headline | details |
|---|---|---|
| Depth stress test | e2e collapses to chance at ~12 layers (vanishing gradients); greedy still learns at layer 40 | [→](docs/RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12) |
| Skip connections (`--skip-input`) | depth finally pays: peak 88.2%@4 → 90.4%@8. DenseNet-style `--skip-all` tested, negative. (Synergizes with width — see scaling track) | [→](docs/RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth) |
| Windowed lookahead (`--window`) | training 2 layers ahead closes ~⅔ of the myopia gap: 90.4% vs e2e's 91.5% (3 seeds), +2.4 pt on MNIST; overlap/receding-horizon variant loses to plain blocks | [→](docs/RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap) |
| Forward-Forward objective (`--objective ff`) | goodness = **popcount** on binary layers, so the whole FF inference is one logic circuit; 2.4 pt behind supervised local CE on digits (86.0%) but **+2.5 pt ahead on MNIST** (76.8%) — and the first lever to exploit depth (17 layers) without skip wiring; needs label-bit replication for sparse random wiring | [→](docs/RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| FF × window, FF negative mining (`--ff-neg`) | windowed lookahead **stacks with FF** (digits 88.0%) unlike with skip; mining works once you warm up before mining (`--ff-neg review --ff-neg-warmup 0.5`): flat on digits but **MNIST 78.2% → 82.0% — the repo's best fixed-budget net**. Pure hard negatives without warm-up collapse. Structured data×label wiring (`--ff-struct`) replaces the label-replication hack at equal MNIST accuracy (tie, not a win) with a tiny pool and zero wasted gates | [→](docs/RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| Identity warm-start (`--warm-start`) | init each layer to reproduce the previous one (ResNet identity block in logic gates), then refine. **Retires the lookahead window**: on plain greedy digits, 94.5% vs the window's 90.4% (3 seeds), and keeps depth productive to 15 layers. Beats the window by ~+11 pt on MNIST too, but single-seed MNIST is noisy (~4 pt) and it stays **below the residual champion** — its value is fixing *plain* greedy, not replacing residual. `--group-boost` (AdaBoost-style reweighting on residual) is a modest +0.43 pt on MNIST | [→](docs/RESULTS.md) |
| Adaptive per-layer epochs (`--epoch-stop` / `--epoch-chain`) | stop each layer when its gate-argmax churn settles, instead of a fixed 120. **Honest negative result**: no variant beats fixed 120 — churn half-decays at ~125 epochs, so 120 was already near-optimal. `--epoch-chain 2` ties it with 5× lower variance. Finding: fully settling a layer *stalls* depth growth (per-layer convergence isn't the goal) | [→](docs/RESULTS.md) |
| Recursion (`--recur`, `--seq`) | weight-tied recursion — `--recur K` iterates a layer K times (parameter compression); `--seq` is RDDLGN-style temporal recurrence (one image row per step, BPTT, `s_t = L([x_t; s_{t-1}])`). Both **collapse from random init and are rescued by identity warm-start**; `--seq` on digits reaches 0.912 (3 seeds) seeing 24 bits/step. Unifying finding: recursing discrete logic needs a near-identity map pointing at *informative* bits (matches RDDLGN's Residual-init requirement). MNIST `--seq` deferred (28-step BPTT too slow interactively) | [→](docs/RESULTS.md) |
**Upstream of the arena — preprocessing, not a learning idea**: input binarization (`--thresholds`). The diagnosis (dead bits, thresholds below quantiles) said *raise* them — the data said the opposite: quantile thresholds lose, **adding a lower plane wins** (MNIST residual 89.96 → 90.71%, 3/3 seeds). Stacked on the then-current residual+skip headline (93.72%), it lifted the 3-seed mean to **94.27%** (+0.55 pt, 3/3 seeds, best single 94.60%, every run bit-exactly verified). But this changes the *input encoding* (it adds a fourth threshold plane, i.e. more input bits), not the network or the learning — a data-representation lever that transfers to any method — so it is held **off the arena** and credited separately from the method ideas. The headline is the method's own 94.64% (that comparison predates the depth-exploration result and hasn't been re-run on top of it). [→](docs/RESULTS.md)

**Scaling track — reference** (width / ensembles / bigger budgets, parked):

| experiment | headline | details |
|---|---|---|
| Memory-matched width | at equal training memory (4× wider layers), greedy **beats** e2e: 95.0% vs 91.5% mean, 3 seeds | [→](docs/SCALING.md#memory-matched-comparison-equal-training-memory-greedy-wins) |
| MNIST first pass | the pattern replicates at 45× the data: memory-matched greedy+skip 84.6% vs e2e 80.1% (absolute numbers far below difflogic-scale budgets, stated honestly) | [→](docs/SCALING.md#mnist-the-pattern-replicates-first-pass-small-budget) |
| Ensemble voting (`--ensemble`) | parallel hard circuits + vote: stacks with every lever (digits 96.4%), but **loses to depth at equal compute and equal gates** — what it buys is latency: the same accuracy at half the critical path, for 1.5× the area | [→](docs/SCALING.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever) |
| MNIST scaling | width is the dominant lever (4,000 gates: 89.8% single) and ensembling stacks: **90.9%** with 4×4,000+skip; more epochs and window×width confirmed dead (+0.1 pt each) | [→](docs/SCALING.md#mnist-scaling-width--ensembles-push-past-90) |

Full run logs (environment, commands, raw output): **one GitHub issue per experiment** ([#1](https://github.com/Mming-Lab/greedy-lgn/issues/1) main run … [#10](https://github.com/Mming-Lab/greedy-lgn/issues/10) Forward-Forward), linked from each [RESULTS.md](docs/RESULTS.md) section.

## Quick start

```bash
pip install torch scikit-learn
# main track (500 gates, single net)
python experiment.py                                      # full run, a few min on CPU
python experiment.py --gates 200 --epochs 30 --max-layers 3   # ~20 s smoke test
python experiment.py --skip-e2e                           # greedy + simplification only
python experiment.py --device cuda                        # same experiment on GPU (~10x faster)
python experiment.py --window 2 --commit 2 --win-loss all # 2-layer lookahead blocks (+2 pt)
python experiment.py --objective ff --ff-label-rep 38 --skip-e2e   # Forward-Forward objective
python experiment.py --objective ff --ff-label-rep 38 --window 2 --commit 2 --ff-neg review --ff-neg-warmup 0.5 --skip-e2e   # best fixed-budget stack
python experiment.py --group-residual --skip-e2e                              # boosting readout (the winner)
python experiment.py --warm-start 5 --max-layers 20 --skip-e2e                 # identity init: depth stays productive
python experiment.py --seq --warm-start 3 --skip-e2e                           # row-sequential recurrence (RDDLGN-style)
# the headlines (MNIST, GPU; --checkpoint lets you stop and resume)
# Vol.1 -- fixed budget, 94.64% (3-seed mean). 8-11 h/seed: the depth search runs to 41-95 layers
python experiment.py --dataset mnist --device cuda --batch 512 --group-residual --skip-input \
  --max-layers 120 --patience 20 --skip-e2e --checkpoint runs/vol1.pt
# Vol.2 -- budget unlocked, 97.04% (3-seed mean, ~66k gates after simplification). ~40 min/seed
python experiment.py --dataset mnist --device cuda --batch 1024 --epochs 60 --gates 8000 \
  --group-residual --skip-input --max-layers 12 --patience 12 --skip-e2e --checkpoint runs/vol2.pt

# scaling track (reference)
python experiment.py --ensemble 4 --skip-e2e              # 4 independent nets + voting
python experiment.py --device cuda --gates 2000 --skip-input --max-layers 16 --skip-e2e --ensemble 4   # digits best with scaling (96.4%)
python tools/vote_checkpoints.py --members runs/a.pt:41 runs/b.pt:78   # vote saved circuits, no retraining
```

## Open questions

- **Does structure buy what width couldn't?** Width stops paying once the net memorizes the training set (Vol. 2's wall). Convolution — weight sharing with a receptive field that grows — is the obvious candidate for generalization instead of memorization. MNIST's 97.5% line is the warm-up; CIFAR-10 is the real test.
- **Why is greedy so gate-hungry?** Layer-local training costs ~1.7× difflogic's gates for the same accuracy band, and ~50% of gates are functionally redundant by diagnosis. Is that inherent to the greedy setting, or just a simplifier that only merges *structural* duplicates?
- [Mono-Forward](https://arxiv.org/abs/2501.09238)-style projection losses.
- [ ] Simplify *between* growth steps (currently done once at the end) and rewire the next layer to the simplified circuit.
- [ ] Export simplified circuits to Verilog / run through ABC for comparison with proper logic synthesis.

## What this borrows, and what it puts together

Almost every ingredient here is from prior work — this section is about being explicit, not claiming credit. The whole repo is organized around one simple recipe:

> *Train one logic layer with a local loss, discretize it immediately, freeze it, and train the next layer on the real 0/1 bits.*

I have **not** surveyed the literature and don't claim this recipe — or any piece of it — is new; it may well exist already. What I can say concretely about each piece, without any novelty claim:

| property | where it comes from |
|---|---|
| No multipliers / DSPs / floats; maps to FPGA LUTs | **Inherited** from LGNs ([difflogic](https://github.com/Felix-Petersen/difflogic)) — not mine, just the platform. |
| **Zero discretization gap** | Follows directly from the recipe (each layer is discretized before the next trains, so the reported accuracy *is* the hard circuit's). I haven't seen this exact setup in the few LGN papers I've looked at, but I haven't searched properly — take that as ignorance, not a claim. Not "the first verified-equals-deployed network" either (exact-by-construction routes exist outside LGNs, e.g. LogicNets' truth-table enumeration). |
| Training memory = one layer, not depth | **Not special** — any greedy layer-wise scheme (Cascade-Correlation, Forward-Forward) has this. |
| Adaptive depth / grow-and-freeze | **Cascade-Correlation heritage (1990)** — old idea. One reading I liked: since circuit depth = critical-path latency, stopping at the accuracy plateau happens to give a low-latency circuit for that accuracy. Post-deployment growth is likewise possible in principle, but my own depth-stress data shows added depth only pays off with skip wiring, so treat it as hand-waving. |
| Windowed lookahead (`--window`) | **Block-wise greedy training exists** (Belilovsky et al., 2019, with auxiliary heads). What I added on top: the blocks are discretized and frozen as they're committed (bit-exact prefix preserved), depth stays adaptive, and I report the overlap ablation (commit < window) — including the negative result that overlap doesn't beat plain blocks. |

## Related work

- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) (Petersen et al., NeurIPS 2022) and [difflogic](https://github.com/Felix-Petersen/difflogic)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732) (NeurIPS 2024) — includes post-training logic synthesis
- [Light Differentiable Logic Gate Networks](https://arxiv.org/abs/2510.03250) (2025) — depth via reparameterization (the backprop-side answer to the same problem)
- [Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](https://arxiv.org/abs/2506.07500) (2025) — progressively discretizes and freezes layers *during* backprop training. Same freeze mechanism, opposite purpose: it freezes to remove the discretization gap while keeping backprop; this repo freezes to remove backprop (and the gap vanishes as a by-product). Gradient boosting (Friedman) is the other close relative — stagewise residual fitting with frozen discrete weak learners (trees) — which is exactly what `--group-residual` does, except the weak learners here stack on each other's output bits instead of all reading the raw input.
- [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) (Hinton, 2022)
- Cascade-Correlation (Fahlman & Lebiere, 1990) — the original "grow and freeze" network
- Greedy layerwise learning can scale to ImageNet (Belilovsky et al., ICML 2019) — block-wise greedy training with auxiliary heads, the closest relative of `--window`
- **I have not done a proper literature search** (just asked an AI), so I make no claims about what is or isn't new. This combination — or any part of it — may already exist under names I don't know; if you know of prior work, please open an issue so I can point to it.

## License

MIT

---

## 日本語概要

論理ゲートネットワーク(DLGN)を**逆伝播なしで1層ずつ**学習する実証実験です。各層をローカルな損失(GroupSum+交差エントロピー)で学習したら**即座に離散化して凍結**し、次の層は本物の0/1ビットの上で学習します。検証精度が頭打ちになったら層の追加を止めるため、深さは自動決定されます。学習後に回路を簡略化し、出力が完全に同一であることをビット単位で検証します。

> **論文も読まない素人がAIと壁打ちしながらのお遊びです。** AIとアイデアを出し合って、実験して、精度(ポイント)の変化を楽しんでいるだけです。査読も受けていませんし、文献調査もAIに聞いた程度なので、新規性や優先権は一切主張しません。ここにあるアイデアの多くは、私が知らない名前で既に存在しているかもしれません（調べていないので、あるとも無いとも言えません）。再現できる遊びのログとして読んでください。もし既存研究と重複していたら、それが普通です — issueで教えてもらえると助かります。

**固定予算から始めて、そこを超えました。**

最初は**500ゲート/層・単発ネット**という自分で課した予算でやっていました。「資源を足したから上がった」ではなく「*アイデア*で上がった」を見たかったからです。そこで**94.64%**まで登り([Vol.1](WHITEPAPER1.md))、そして分かったのは**天井は手法ではなく予算だった**ということ。予算を外したら(レシピは同じ、ノブ1つ)、先行研究のコンパクト設計点(95.8%)を**18分で抜き**、**MNIST 97.04%**に到達しました([Vol.2](WHITEPAPER2.md)。3シード、8,000ゲート×深さ10、簡略化後約66,000ゲート、全ランビット等価検証済み、6GBのノートGPUで1本40分)。その先には**暗記の壁**(train=100%でtestが97.0%に凍結)がありました。

つまり看板は次のステージへ移りました。ただし**固定予算は引退したのではなく、物差しに昇格しました** — 新しいアイデアが本物かノイズかを見分けるには今も一番速い場所です(ダメなアイデアはdigitsで数秒、MNISTでも1時間で決着)。下の実験一覧は全部この物差しで測っています。変わったのは「そこが天井ではなくなった」ことだけです。

**正直な負けも先に**: ゲート効率では勾配学習の先行研究に明確に負けています(difflogic 48,000ゲートで97.69%、LILogic Net 8,000ゲートで98.45%、対してこちらは約66,000ゲートで97.04%)。面白いのは効率ではなく、**層をまたぐ逆伝播を一切使わずにこの帯まで登れたこと**です。

以下は物差し(固定予算)の上での話。MNISTを審判にした階段は **素74.3% → 先読み窓76.6% → FF+窓+誤答復習82.0% → 残差readout(`--group-residual`)90.9% → 残差+skip 94.5%**(単発500ゲートのリポジトリ記録・看板。3シード平均94.64、全ラン**ビット等価検証済み**)。**明確な勝者は残差readout**です — 素のgreedyは各層の答えを捨てて最終層だけで答えるせいで深さとともに劣化しますが、各層のクラス予測を累積する(=素朴なブースティング)だけで劣化が消えます。看板が93.72→94.64%に上がったのは**深さ探索の延長**(`--patience 10`)で、既定のpatience=2が同じchampを深さ26-34で止めていたのを80まで探させただけ(+0.81pt、3シード全勝。留保: 深く探すほどテスト集合で深さを選ぶバイアスも増える)。さらに**上流の入力二値化**(土俵の外)で「低い閾値の面を足す」(`--thresholds`)と残差単体90.0→90.7%(3シード全勝)、旧看板93.72%に重ねたときは3シード平均**94.27%**(+0.55pt、ベスト単発94.60%、検証済み)。ただしこれは入力の符号化を変える前処理(面を1枚足す=入力ビット増)でどの手法にも効くので**土俵の外**に置く(深い新看板の上での再測定は未実施)。詳細な表は英語本文の「The arena」を参照。

**資源を使う側**: 幅(`--gates`)を上げるのは「良いアイデア」ではなく面積を払って精度を買う行為なので、上のアイデア表からは外してあります。ただし「賢くない」と「つまらない」は別で、そのノブ1つを回した記録が[Vol.2](WHITEPAPER2.md)(94.64→**97.04%**)です。**アンサンブル**(`--ensemble`)はさらに別枠 — 実測すると計算量でもゲート数でも深さに負け(3ネット×深さ40=94.86% vs 1ネット×深さ78=94.91%)、本当に買えるのは**レイテンシ**(同じ精度を半分の臨界パスで、面積1.5倍)でした。詳細は[SCALING.md](docs/SCALING.md)。

**同じ単発500の土俵で、逆伝播(e2e)と比べると:** 素のgreedyは出発点では逆伝播に負けます(digits 88.2 vs 93.6、MNIST 74.3 vs 81.8)。でもレバーを入れると逆転し、データが大きいほど差が開きます — **digits +2.8pt(96.4 vs 93.6)、MNIST +12.8pt(94.6 vs 81.8)**。理由は深さ: 逆伝播は勾配消失で深さ6あたりが頭打ち(単発500のe2eは81.8%@6でピーク→以降崩壊)、greedyは層またぎの勾配が無いので40〜95層まで深さを使える。加えて離散化ギャップが構造的にゼロ・学習メモリが1層分、という利点もあります。

**実験一覧(本線=固定予算のアイデア勝負)** — 詳細は各リンク先:

| 実験(フラグ) | 一言でいうと | 結果 |
|---|---|---|
| 残差readout(`--group-residual`) | 各層の答えを捨てずに積み上げていく(素朴なブースティング)。「答えの積み重ね=学習」 | **MNIST 74.3→90.9%、現在の最高記録構成の土台** [→](docs/RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes) |
| 深さ耐性テスト | 何層まで学習できるか力比べ(アイデアでなく診断) | 逆伝播は12層で崩壊、greedyは40層でも学べる(精度のピークは別問題) [→](docs/RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12) |
| skip配線(`--skip-input`) | どの層にも元画像を見せ直す配線 | 深さで劣化しなくなる。88.2→90.4%、最高記録構成に+3pt寄与 [→](docs/RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth) |
| 先読み窓(`--window`) | 1層ずつでなく2層先まで見てから確定(近視の緩和) | +2pt — 今はwarm-startに役目を譲った [→](docs/RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap) |
| Forward-Forward(`--objective ff`) | 「正しいラベルを重ねた画像では発火を増やし、偽ラベルでは減らす」学習。推論まで純論理回路 | MNISTで素より+2.5pt [→](docs/RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| 誤答の重点復習(`--ff-neg`) | まず普通に学習→模試→間違えた問題を重点復習(人間の勉強法と同じ発想) | MNIST 82.0%(残差以前の最高値) [→](docs/RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| 恒等warm-start(`--warm-start`) | 新しい層をゼロから作らず「前の層の完コピ」から微調整で始める | 先読み窓を+4pt圧倒して引退させた [→](docs/RESULTS.md) |
| 適応エポック(`--epoch-stop`等) | 各層の学習を固定120回でなく、変化が収まった時点で自動で打ち切る | **負け**: 固定120が偶然ほぼ最適だった(churn半減点が約125回) [→](docs/RESULTS.md) |
| 再帰(`--recur` / `--seq`) | 同じ層を使い回す/画像を1行ずつ流して「記憶」で読む | 恒等初期化がないと崩壊、恒等ありでseqはdigits 91.2% [→](docs/RESULTS.md) |
| 畳み込み配線(`--local` / `--conv`、進行中) | 近くの画素だけ見る配線 → 重み共有カーネル+プーリングの本物の畳み込みへ | 局所配線単体はMNISTで負け、本物の畳み込みはdigitsで密と同点。MNIST審判はメモリ律速で保留(28×28がVRAM超) [→](docs/RESULTS.md) |

**本線の外のレバー**(学習アイデアの勝負とは別枠):
- **前処理 — 入力二値化(`--thresholds`)**: 画素を白黒に割るしきい値の調整。薄い筆致を拾う低いしきい値(面)を足すのが正解で、残差単体で+0.75pt。当時の看板(残差+skipの93.72%、3シード平均)に重ねたときは3シード平均**94.27%**(+0.55pt、3/3勝、検証済み)。ただしこれは学習でもネットワークでもなく**入力の符号化を変える**(面を1枚足す=入力ビットを増やす)前処理でどの手法にも効くので、**土俵の外**に置き手法の手柄とは区別して数える(看板は手法単体の94.64%。深さ延長後の看板に重ねた再測定は未実施) [→](docs/RESULTS.md)
- **スケーリング**(参考、[SCALING.md](docs/SCALING.md)): 計算資源=推論回路面積で精度を買う側。メモリ等価比較(学習メモリを揃えるとgreedyがe2eに全勝 95.0 vs 91.5)/アンサンブル投票(`--ensemble`、独立回路を並べて多数決 — digits 96.4%)/MNISTスケーリング(幅が支配的、90.9%)

**その他の死にレバー**(正直な記録 — いずれも既存フラグの設定変更・組合せで実測した負け): エポック増(`--epochs`2倍で+0.1pt)、window×幅・window×skip(組合せが加算されない)、warmupなしのhard負例(`--ff-neg hard`単体は崩壊)、分位点閾値(`--thresholds q3〜q5`は負け — 診断は「上げろ」、データは「低い面を足せ」だった)、FF-Residual(`--objective ff-residual`、ラベル埋め込み無しの1パスFF+残差累積 — CE残差に digits 3シードで一貫して-1pt前後。boostでも、FFで最大レバーだった誤答復習の移植(`--group-review`)でも埋まらず、「CEの暗黙の負例集中を手放したのが原因」という仮説は**反証**された。負ける理由は未解明)、深さカーブの外挿による早期打ち切り(飽和カーブ自体はよく当てはまる(Hill型 R²≥0.998)が、15層フィットの予測が3シード目で+1.9pt外れて実用にならない)。

各実験のセットアップ・数値表・**反証された仮説**は [RESULTS.md](docs/RESULTS.md)(スケーリング系は [SCALING.md](docs/SCALING.md))に、生ログは実験ごとの個別issue(#1〜#14、各セクションからリンク)にあります。回路の中身を覗く診断ツール([tools/diagnose.py](tools/diagnose.py)=ゲート種類分布・機能的冗長度、[tools/dynamics.py](tools/dynamics.py)=学習済み再帰セルの発振器census)もあります。

位置づけ: 構成要素のほとんどは先行研究からの借り物です。全体は「**各層を学習→即離散化→凍結し、次層を本物のビット上で学習する**」という素朴なレシピで組み立てられていますが、これが新しいかどうかは分かりません(ちゃんと調べていないので既出の可能性は高いです)。離散化ギャップゼロはこのレシピの帰結、メモリ効率と適応深さはCascade-Correlation / Forward-Forward由来です。詳細は英語本文の「What this borrows, and what it puts together」を参照してください。
