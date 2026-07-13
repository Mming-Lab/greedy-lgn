# greedy-lgn

**Backpropagation-free, layer-by-layer training of Differentiable Logic Gate Networks** — with immediate discretization, adaptive depth, and incremental logic simplification.

> Train one logic layer at a time with a local loss, discretize it, freeze it, and let the next layer learn on *real* 0/1 bits. Stop adding layers when accuracy plateaus. Simplify the circuit as it grows.

Proof-of-concept. Runs on CPU in a few minutes. A single self-contained script, no dependencies beyond `torch` and `scikit-learn`.

📄 **Narrative write-up:** [WHITEPAPER.md](WHITEPAPER.md) — the climb to 94% as a reproducible experiment log (with a Japanese abstract).

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

## The arena: 500 gates/layer, one network

Everything on the main track runs at a **fixed budget — 500 gates/layer, single network** — because the game here is watching which *ideas* move the accuracy points, not how much compute gets spent. The starting point, on `sklearn` digits (8×8, thermometer-binarized to 192 bits), CPU:

| | greedy (this repo) | end-to-end backprop |
|---|---|---|
| depth | **4 (chosen automatically)** | 4 (copied from greedy) |
| hard-circuit test accuracy | 88.2% | **93.6%** |
| discretization gap | **0 (by construction)** | 0.0 at this scale |
| float logits held during training | **8,000 (one layer)** | 32,000 (×4) |
| circuit after simplification | 2,000 → **1,316 gates (65.8%)**, bit-identical | — |

Plain local training loses ~5 pt of accuracy to backprop — in exchange for zero discretization gap, ~1/depth training memory, automatic depth, and a simplifiable circuit. From that start, the idea ladder so far (hard-circuit test accuracy, gap still structurally zero throughout):

| idea (500 gates, single net) | digits (3 seeds) | MNIST |
|---|---|---|
| plain greedy (GroupSum + CE) | 88.4% | 74.3% |
| + skip-input (`--skip-input`) | 89.1% | —¹ |
| + windowed lookahead (`--window 2 --commit 2`) | 90.4% | 76.6% |
| Forward-Forward objective (`--objective ff`) | 86.0% | 76.8% |
| FF + window | 88.0% | 78.2% |
| FF + window + hard-negative mining (`--ff-neg`) | 89.7%² | 82.0%³ |
| **residual/boosting readout (`--group-residual`)** | **96.4%** | **90.9%** |
| **residual + skip-input** | **97.3%** | **93.9%⁴** |
| **residual + skip + low-plane binarization (`--thresholds`)** | — | **94.3%⁵** |

¹ skip alone is a depth/width-synergy lever — modest at 500-gate single net, and it actively *hurts* FF, so it isn't a fixed-budget winner. ² `mix` negatives. ³ `review` + 0.5 warm-up. ⁴ at depth 38 (still slowly climbing); residual-alone 90.9% converges at depth 9. ⁵ **94.27% (3-seed mean, best single 94.60%)** vs a same-protocol control **93.72%** — the low plane wins on **3/3 seeds** (+0.26/+1.12/+0.26 pt). Every run is **bit-exactly verified** through simplification (`identical=True`). It is the repo record; the depth stays ~27–40 (seed 2 was still climbing at the cap), so unlike the seed-1 draw it is not consistently shallower — the accuracy gain is the robust claim.

**The clear winner is the residual readout.** Plain greedy throws away every layer's class prediction except the last, which is exactly why accuracy decays with depth (shallow layers see fresh image info, but their good answers are discarded). Accumulate each layer's prediction instead — plain boosting — and the decay vanishes: MNIST climbs from 74.3% to **90.9%** (single 500-gate net, no skip/window/ensemble, not overfitting: train 91.9 / test 90.6). That already matches the scaling-track flagship below (4,000 gates × 4 ensemble) at a fraction of the area. Stacking `--skip-input` on top — residual accumulates the *answer*, skip re-exposes the *image*, different cures for the same decay — compounds to **93.9%**, a new repo record, still at 500 gates single net (honest cost: depth grows 9 → 38, i.e. more latency). The readout is plain boosting / deep supervision — the observation here is just that it fixes greedy-LGN's depth decay cleanly. [→](RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes)

## Off the main track: scaling levers

Wider layers (`--gates`) and ensembles (`--ensemble`) buy accuracy by spending compute/area, not by being good ideas — so they live outside the arena, in **[SCALING.md](SCALING.md)** (parked reference: digits 96.4% / MNIST 90.9%). Note the residual readout above already matches that MNIST flagship as a single 500-gate net.

## All experiments (details in [RESULTS.md](RESULTS.md))

**Main track — fixed-budget ideas** (500 gates/layer, single net):

| experiment | headline | details |
|---|---|---|
| Depth stress test | e2e collapses to chance at ~12 layers (vanishing gradients); greedy still learns at layer 40 | [→](RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12) |
| Skip connections (`--skip-input`) | depth finally pays: peak 88.2%@4 → 90.4%@8. DenseNet-style `--skip-all` tested, negative. (Synergizes with width — see scaling track) | [→](RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth) |
| Windowed lookahead (`--window`) | training 2 layers ahead closes ~⅔ of the myopia gap: 90.4% vs e2e's 91.5% (3 seeds), +2.4 pt on MNIST; overlap/receding-horizon variant loses to plain blocks | [→](RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap) |
| Forward-Forward objective (`--objective ff`) | goodness = **popcount** on binary layers, so the whole FF inference is one logic circuit; 2.4 pt behind supervised local CE on digits (86.0%) but **+2.5 pt ahead on MNIST** (76.8%) — and the first lever to exploit depth (17 layers) without skip wiring; needs label-bit replication for sparse random wiring | [→](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| FF × window, FF negative mining (`--ff-neg`) | windowed lookahead **stacks with FF** (digits 88.0%) unlike with skip; mining works once you warm up before mining (`--ff-neg review --ff-neg-warmup 0.5`): flat on digits but **MNIST 78.2% → 82.0% — the repo's best fixed-budget net**. Pure hard negatives without warm-up collapse. Structured data×label wiring (`--ff-struct`) replaces the label-replication hack at equal MNIST accuracy (tie, not a win) with a tiny pool and zero wasted gates | [→](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| Identity warm-start (`--warm-start`) | init each layer to reproduce the previous one (ResNet identity block in logic gates), then refine. **Retires the lookahead window**: on plain greedy digits, 94.5% vs the window's 90.4% (3 seeds), and keeps depth productive to 15 layers. Beats the window by ~+11 pt on MNIST too, but single-seed MNIST is noisy (~4 pt) and it stays **below the residual champion** — its value is fixing *plain* greedy, not replacing residual. `--group-boost` (AdaBoost-style reweighting on residual) is a modest +0.43 pt on MNIST | [→](RESULTS.md) |
| Adaptive per-layer epochs (`--epoch-stop` / `--epoch-chain`) | stop each layer when its gate-argmax churn settles, instead of a fixed 120. **Honest negative result**: no variant beats fixed 120 — churn half-decays at ~125 epochs, so 120 was already near-optimal. `--epoch-chain 2` ties it with 5× lower variance. Finding: fully settling a layer *stalls* depth growth (per-layer convergence isn't the goal) | [→](RESULTS.md) |
| Recursion (`--recur`, `--seq`) | weight-tied recursion — `--recur K` iterates a layer K times (parameter compression); `--seq` is RDDLGN-style temporal recurrence (one image row per step, BPTT, `s_t = L([x_t; s_{t-1}])`). Both **collapse from random init and are rescued by identity warm-start**; `--seq` on digits reaches 0.912 (3 seeds) seeing 24 bits/step. Unifying finding: recursing discrete logic needs a near-identity map pointing at *informative* bits (matches RDDLGN's Residual-init requirement). MNIST `--seq` deferred (28-step BPTT too slow interactively) | [→](RESULTS.md) |
**Upstream of the arena — preprocessing, not a learning idea**: input binarization (`--thresholds`). The diagnosis (dead bits, thresholds below quantiles) said *raise* them — the data said the opposite: quantile thresholds lose, **adding a lower plane wins** (MNIST residual 89.96 → 90.71%, 3/3 seeds, and it feeds the verified 94.27% 3-seed record). A data-representation lever that transfers to any method, so it is credited separately from the method ideas. [→](RESULTS.md)

**Scaling track — reference** (width / ensembles / bigger budgets, parked):

| experiment | headline | details |
|---|---|---|
| Memory-matched width | at equal training memory (4× wider layers), greedy **beats** e2e: 95.0% vs 91.5% mean, 3 seeds | [→](SCALING.md#memory-matched-comparison-equal-training-memory-greedy-wins) |
| MNIST first pass | the pattern replicates at 45× the data: memory-matched greedy+skip 84.6% vs e2e 80.1% (absolute numbers far below difflogic-scale budgets, stated honestly) | [→](SCALING.md#mnist-the-pattern-replicates-first-pass-small-budget) |
| Ensemble voting (`--ensemble`) | parallel hard circuits + vote: stacks with everything (digits **96.4% — repo best**); on MNIST 4×500-gate members beat the single 2,000-gate best at **half the training memory**; not a substitute for direct width | [→](SCALING.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever) |
| MNIST scaling | width is the dominant lever (4,000 gates: 89.8% single) and ensembling stacks: **90.9%** with 4×4,000+skip; more epochs and window×width confirmed dead (+0.1 pt each); 8,000 gates OOMs at 6 GB (pools, not eval temporaries) | [→](SCALING.md#mnist-scaling-width--ensembles-push-past-90) |

Full run logs (environment, commands, raw output): **one GitHub issue per experiment** ([#1](https://github.com/Mming-Lab/greedy-lgn/issues/1) main run … [#10](https://github.com/Mming-Lab/greedy-lgn/issues/10) Forward-Forward), linked from each [RESULTS.md](RESULTS.md) section.

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
# scaling track (reference)
python experiment.py --ensemble 4 --skip-e2e              # 4 independent nets + voting
python experiment.py --device cuda --gates 2000 --skip-input --max-layers 16 --skip-e2e --ensemble 4   # digits best with scaling (96.4%)
python experiment.py --dataset mnist --device cuda --batch 4096 --epochs 30 --gates 2000 --skip-input --max-layers 10 --skip-e2e   # MNIST (GPU recommended)
```

## Roadmap / open questions

- [x] **Depth stress test** — backprop dies at ~12 layers, greedy survives 40 ([details](RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12)).
- [x] **Memory-matched comparison** — greedy wins at equal training memory ([details](SCALING.md#memory-matched-comparison-equal-training-memory-greedy-wins)).
- [x] **Skip connections** — `--skip-input` makes depth useful; `--skip-all` negative ([details](RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth)).
- [x] **MNIST first pass** — pattern replicates; absolute accuracy still small-budget ([details](SCALING.md#mnist-the-pattern-replicates-first-pass-small-budget)).
- [x] **Windowed lookahead** — `--window 2` recovers most of the myopia deficit; window > 2 and overlapping commits don't help ([details](RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap)).
- [x] **Ensemble voting** — `--ensemble M` stacks with every other lever; repo best on digits (96.4%) and the training-memory-free path to MNIST scaling ([details](SCALING.md#ensemble-voting-parallel-circuits-are-the-training-memory-free-width-lever)).
- [x] **MNIST scaling, first round** — width × ensembles reaches 90.9%; epochs and window×width are dead ends ([details](SCALING.md#mnist-scaling-width--ensembles-push-past-90)).
- [x] **Forward-Forward objective** — popcount goodness works: behind supervised local CE on digits, ahead on MNIST, and exploits depth without skip wiring ([details](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist)).
- [ ] [Mono-Forward](https://arxiv.org/abs/2501.09238)-style projection losses; better input binarization (fixed-budget friendly).
- [ ] *(parked, scaling track)* MNIST absolute accuracy: 8,000-gate layers (needs the pool-memory fix), FF × width/ensemble, convolutional wiring, CIFAR-10 on [difflogic](https://github.com/Felix-Petersen/difflogic) CUDA kernels.
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
- [The Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345) (Hinton, 2022)
- Cascade-Correlation (Fahlman & Lebiere, 1990) — the original "grow and freeze" network
- Greedy layerwise learning can scale to ImageNet (Belilovsky et al., ICML 2019) — block-wise greedy training with auxiliary heads, the closest relative of `--window`
- **I have not done a proper literature search** (just asked an AI), so I make no claims about what is or isn't new. This combination — or any part of it — may already exist under names I don't know; if you know of prior work, please open an issue so I can point to it.

## License

MIT

---

## 日本語概要

論理ゲートネットワーク(DLGN)を**逆伝播なしで1層ずつ**学習する実証実験です。各層をローカルな損失(GroupSum+交差エントロピー)で学習したら**即座に離散化して凍結**し、次の層は本物の0/1ビットの上で学習します。検証精度が頭打ちになったら層の追加を止めるため、深さは自動決定されます。学習後に回路を簡略化し、出力が完全に同一であることをビット単位で検証します。

> **論文も読まない素人がAIと壁打ちしながらのお遊びです。** AIとアイデアを出し合って、実験して、精度(ポイント)の変化を楽しんでいるだけです。査読も受けていませんし、文献調査もAIに聞いた程度なので、新規性や優先権は一切主張しません。ここにあるアイデアの多くは、私が知らない名前で既に存在しているはずです。再現できる遊びのログとして読んでください。もし既存研究と重複していたら、それが普通です — issueで教えてもらえると助かります。

本線は**500ゲート/層・単発ネットの固定予算**です。この遊びの主役は「計算資源を増やさずにアイデアだけでポイントがどれだけ動くか」で、MNISTを審判にした現在の階段は **素74.3% → 先読み窓76.6% → FF+窓+誤答復習82.0% → 残差readout(`--group-residual`)90.9% → 残差+skip 93.9% → +低側面二値化 94.3%**(単発500ゲートのリポジトリ記録。94.3%は3シード平均・ベスト単発94.60%で、対照93.72%に3/3勝、全ラン**ビット等価検証済み**の看板です)。**明確な勝者は残差readout**です — 素のgreedyは各層の答えを捨てて最終層だけで答えるせいで深さとともに劣化しますが、各層のクラス予測を累積する(=素朴なブースティング)だけで劣化が消えます。さらに上流の入力二値化で「低い閾値の面を足す」(`--thresholds`)と残差90.0→90.7%(3シード全勝)。詳細な表は英語本文の「The arena」を参照。

**スケーリングレバー**(幅=`--gates` とアンサンブル=`--ensemble`)は別トラック(参考、[SCALING.md](SCALING.md))です。計算資源=推論回路面積を突っ込めば確実に精度を買えますが、アイデアの良し悪しは分かりません。参考値: digits 96.4%(2,000ゲート+skip+×4多数決)、MNIST 90.9%(4,000ゲート+skip+×4 soft vote)。**残差readoutは単発500ゲートでこのMNIST旗艦に並び、+skipで超えました**。このトラックは休止中です。

素のgreedyはend-to-end逆伝播に約5pt負けます(88.2% vs 93.6%)が、代わりに離散化ギャップが構造的にゼロ・学習メモリが深さ分の1・深さの自動決定という利点があります。

**実験一覧(本線=固定予算のアイデア勝負)** — 詳細は各リンク先:

| 実験(フラグ) | 一言でいうと | 結果 |
|---|---|---|
| 残差readout(`--group-residual`) | 各層の答えを捨てずに積み上げていく(素朴なブースティング)。「答えの積み重ね=学習」 | **MNIST 74.3→90.9%、現在の最高記録構成の土台** [→](RESULTS.md#residualboosting-readout-accumulate-the-answer-and-the-depth-decay-vanishes) |
| 深さ耐性テスト | 何層まで学習できるか力比べ(アイデアでなく診断) | 逆伝播は12層で崩壊、greedyは40層でも学べる(精度のピークは別問題) [→](RESULTS.md#depth-stress-test-greedy-survives-40-layers-backprop-dies-at-12) |
| skip配線(`--skip-input`) | どの層にも元画像を見せ直す配線 | 深さで劣化しなくなる。88.2→90.4%、最高記録構成に+3pt寄与 [→](RESULTS.md#skip-connections-re-exposing-the-input-turns-survivable-depth-into-usable-depth) |
| 先読み窓(`--window`) | 1層ずつでなく2層先まで見てから確定(近視の緩和) | +2pt — 今はwarm-startに役目を譲った [→](RESULTS.md#windowed-lookahead-training-two-layers-ahead-closes-most-of-the-myopia-gap) |
| Forward-Forward(`--objective ff`) | 「正しいラベルを重ねた画像では発火を増やし、偽ラベルでは減らす」学習。推論まで純論理回路 | MNISTで素より+2.5pt [→](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| 誤答の重点復習(`--ff-neg`) | まず普通に学習→模試→間違えた問題を重点復習(人間の勉強法と同じ発想) | MNIST 82.0%(残差以前の最高値) [→](RESULTS.md#forward-forward-objective-popcount-goodness--behind-on-digits-ahead-on-mnist) |
| 恒等warm-start(`--warm-start`) | 新しい層をゼロから作らず「前の層の完コピ」から微調整で始める | 先読み窓を+4pt圧倒して引退させた [→](RESULTS.md) |
| 適応エポック(`--epoch-stop`等) | 各層の学習を固定120回でなく、変化が収まった時点で自動で打ち切る | **負け**: 固定120が偶然ほぼ最適だった(churn半減点が約125回) [→](RESULTS.md) |
| 再帰(`--recur` / `--seq`) | 同じ層を使い回す/画像を1行ずつ流して「記憶」で読む | 恒等初期化がないと崩壊、恒等ありでseqはdigits 91.2% [→](RESULTS.md) |
| 畳み込み配線(`--local` / `--conv`、進行中) | 近くの画素だけ見る配線 → 重み共有カーネル+プーリングの本物の畳み込みへ | 局所配線単体はMNISTで負け、本物の畳み込みはdigitsで密と同点。MNIST審判はメモリ律速で保留(28×28がVRAM超) [→](RESULTS.md) |

**本線の外のレバー**(学習アイデアの勝負とは別枠):
- **前処理 — 入力二値化(`--thresholds`)**: 画素を白黒に割るしきい値の調整。薄い筆致を拾う低いしきい値を足すのが正解で、+0.75pt・**最高記録94.3%(3シード平均)に寄与**。学習の工夫ではなくデータ表現の改善なので、どの手法にも効く=手法の手柄とは区別して数える [→](RESULTS.md)
- **スケーリング**(参考、[SCALING.md](SCALING.md)): 計算資源=推論回路面積で精度を買う側。メモリ等価比較(学習メモリを揃えるとgreedyがe2eに全勝 95.0 vs 91.5)/アンサンブル投票(`--ensemble`、独立回路を並べて多数決 — digits 96.4%)/MNISTスケーリング(幅が支配的、90.9%)

**その他の死にレバー**(正直な記録 — いずれも既存フラグの設定変更・組合せで実測した負け): エポック増(`--epochs`2倍で+0.1pt)、window×幅・window×skip(組合せが加算されない)、warmupなしのhard負例(`--ff-neg hard`単体は崩壊)、分位点閾値(`--thresholds q3〜q5`は負け — 診断は「上げろ」、データは「低い面を足せ」だった)。

各実験のセットアップ・数値表・**反証された仮説**は [RESULTS.md](RESULTS.md)(スケーリング系は [SCALING.md](SCALING.md))に、生ログは実験ごとの個別issue(#1〜#11、各セクションからリンク)にあります。回路の中身を覗く診断ツール([diagnose.py](diagnose.py)=ゲート種類分布・機能的冗長度、[dynamics.py](dynamics.py)=学習済み再帰セルの発振器census)もあります。

位置づけ: 構成要素のほとんどは先行研究からの借り物です。全体は「**各層を学習→即離散化→凍結し、次層を本物のビット上で学習する**」という素朴なレシピで組み立てられていますが、これが新しいかどうかは分かりません(ちゃんと調べていないので既出の可能性は高いです)。離散化ギャップゼロはこのレシピの帰結、メモリ効率と適応深さはCascade-Correlation / Forward-Forward由来です。詳細は英語本文の「What this borrows, and what it puts together」を参照してください。
