# greedy-lgn

**Layer-by-layer training of Differentiable Logic Gate Networks** — train one logic layer at a time with a local loss, discretize it, freeze it, and let the next layer learn on *real* 0/1 bits. Stop adding layers when accuracy plateaus. Simplify the circuit as it grows.

> **Just me playing around with logic gates, not research.** I don't read papers — I bounce ideas off an AI assistant, run the experiments, and enjoy watching the accuracy points move. That's the whole thing. Nothing here is peer-reviewed, and the only "literature search" was asking the AI, so I make **no novelty or priority claims**. Plenty of these ideas probably already exist under names I don't know. Read it as a reproducible experiment log, not a contribution; if it duplicates prior work, that's expected — pointers are welcome via an issue.

Proof-of-concept. Runs on CPU in a few minutes. A single self-contained script, no dependencies beyond `torch` and `scikit-learn`. Full experiment log with every number: **[docs/RESULTS.md](docs/RESULTS.md)**.

**A few numbers, for context** (MNIST, hard circuit, bit-exact verified). These are *not* a leaderboard entry — they are where my tinkering happened to land, next to the published work I was reading about:

| | accuracy | gates | training |
|---|---|---|---|
| this repo, 500 gates/layer | 94.6% (3 seeds) | ~40k | layer-local, window 1 |
| this repo, 8,000 gates/layer | 97.1% (3 seeds) | ~86k | layer-local, window 1 |
| this repo, 3 nets voting | 97.6% | ~200k | ensemble of the above |
| [difflogic](https://arxiv.org/abs/2210.08277) small (e2e backprop) | 97.69% | 48k | end-to-end |
| [LILogic Net](https://arxiv.org/abs/2511.12340) (gradient-based) | 98.45% | 8k | end-to-end, learns wiring |

**On gate efficiency this loses clearly, and that's fine** — it's a hobby log, not a competitor. The part I find interesting is the constraint: how far a purely *layer-local* recipe (no gradient crossing a frozen layer) can be pushed. The trained circuits are downloadable ([checkpoints](https://github.com/Mming-Lab/greedy-lgn/releases/tag/vol2-8000-checkpoints)): re-verify them in minutes, no training needed.

> **A note on "backprop-free":** with the default `--window 1`, no gradient ever crosses a layer — each layer is trained alone, frozen, done. Some experiments below use `--window W` (train W layers jointly, freeze one), which *does* backpropagate — but only within the W-layer window, never into the frozen prefix. Results are labelled accordingly. See [the precise accounting](docs/RESULTS.md#how-much-backprop-is-actually-used-2026-07-21).

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

## Comparing ideas at a fixed budget

Most of the experiments below fix the budget at **500 gates/layer, one net**, and vary one idea at a time. Not because 500 is special — it isn't — but because it's a fast, cheap bench: a bad idea dies on digits in seconds, and MNIST settles it within the hour. Bigger budgets come later in the log; they buy accuracy by spending area, which is a different (and less interesting to me) kind of result.

The one number that captures the fixed-budget climb (MNIST, 500 gates/layer, single net, hard circuit, gap structurally zero throughout):

> plain greedy **74.3%** → +windowed lookahead **76.6%** → +FF & hard-negative review **82.0%** → **residual readout 90.9%** → +skip-input **94.6%** (3-seed mean 94.64, `--patience 10`, bit-exact)

**The clear winner is the residual readout.** Plain greedy throws away every layer's class prediction except the last, which is why accuracy decays with depth. Accumulate each layer's prediction instead — plain boosting — and the decay vanishes (74.3 → 90.9%); adding `--skip-input` (residual accumulates the *answer*, skip re-exposes the *image*) compounds to 94.6%. Full ladder, every lever's win *and* loss, and the greedy-vs-backprop comparison (greedy loses at the plain start but overtakes by +12.8 pt once levers are on, because backprop's gradients vanish past ~6 layers while greedy stays productive to 40–80): **[RESULTS.md](docs/RESULTS.md)**.

## Beyond the fixed budget

Everything above holds the budget at 500 gates so a lever's win is about the lever. Spending area instead:

- **Width** (`--gates 8000`): 94.6 → **97.1%** (3-seed mean, ~86k gates), then a **memorization wall** (train = 100%, test frozen ~97.0%). [→](docs/RESULTS.md#width-unlocked-8000-gateslayer-the-memorization-wall)
- **Ensemble** (vote 3 saved nets, zero extra training): **97.6%** — the highest number here — but ~200k gates and 3× inference. [→](docs/SCALING.md)
- **On gate efficiency this loses** to gradient-trained work (difflogic 48k/97.69%, LILogic 8k/98.45%). Stated plainly, not hidden.

Every experiment — main levers, negatives, diagnostics, the width wall, the backprop accounting — lives in **[RESULTS.md](docs/RESULTS.md)** (scaling levers in **[SCALING.md](docs/SCALING.md)**), each linked to its raw-log GitHub issue. New experiments go there, not here.

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
# the bigger MNIST runs (GPU; --checkpoint lets you stop and resume)
# 500 gates/layer -- 94.6% (3-seed mean). ~1.5 h/seed: the depth search runs to 40-110 layers
python experiment.py --dataset mnist --device cuda --batch 512 --group-residual --skip-input \
  --max-layers 120 --patience 20 --skip-e2e --checkpoint runs/mnist500.pt
# 8,000 gates/layer -- 97.1% (3-seed mean, ~86k gates after simplification). ~40 min/seed
python experiment.py --dataset mnist --device cuda --batch 1024 --epochs 60 --gates 8000 \
  --group-residual --skip-input --max-layers 12 --patience 12 --skip-e2e --checkpoint runs/mnist8000.pt

# scaling track (reference)
python experiment.py --ensemble 4 --skip-e2e              # 4 independent nets + voting
python experiment.py --device cuda --gates 2000 --skip-input --max-layers 16 --skip-e2e --ensemble 4   # digits best with scaling (96.4%)
python tools/vote_checkpoints.py --members runs/a.pt:41 runs/b.pt:78   # vote saved circuits, no retraining
```

## Open questions

- **Does structure buy what width couldn't?** Width stops paying once the net memorizes the training set (the [memorization wall](docs/RESULTS.md#width-unlocked-8000-gateslayer-the-memorization-wall)). Convolution — weight sharing with a receptive field that grows — is the obvious candidate for generalization instead of memorization. CIFAR-10 would be the real test.
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

論理ゲートネットワーク(DLGN)を**1層ずつ層ローカルに**学習する実証実験です。各層をローカルな損失(GroupSum+交差エントロピー)で学習したら**即座に離散化して凍結**し、次の層は本物の0/1ビットの上で学習します。凍結済みの層には勾配が入りません(既定`--window 1`では層をまたぐ逆伝播はゼロ。`--window W`のときだけW層ぶん逆伝播しますが、凍結部分には決して入りません)。検証精度が頭打ちになったら層の追加を止めるため、深さは自動決定されます。学習後に回路を簡略化し、出力がビット単位で同一であることを検証します。

> **論文も読まない素人がAIと壁打ちしながらのお遊びです。** AIとアイデアを出し合って、実験して、精度(ポイント)の変化を楽しんでいるだけです。査読も受けていませんし、文献調査もAIに聞いた程度なので、新規性や優先権は一切主張しません。ここにあるアイデアの多くは、私が知らない名前で既に存在しているかもしれません（調べていないので、あるとも無いとも言えません）。再現できる遊びのログとして読んでください。もし既存研究と重複していたら、それが普通です — issueで教えてもらえると助かります。

多くの実験は**500ゲート/層・単発ネット**という固定予算で回しています。500が特別だからではなく、速くて安い実験台だからです(ダメなアイデアはdigitsで数秒、MNISTでも1時間で決着)。MNISTを審判にしたアイデアの階段:

> 素**74.3%** → 先読み窓**76.6%** → FF+誤答復習**82.0%** → **残差readout(`--group-residual`)90.9%** → +skip **94.6%**(3シード平均94.64、全ラン**ビット等価検証済み**)

**明確な勝者は残差readout**です — 素のgreedyは各層の答えを捨てて最終層だけで答えるせいで深さとともに劣化しますが、各層のクラス予測を累積する(=素朴なブースティング)だけで劣化が消えます(74.3→90.9%)。ここに元画像を見せ直すskipを重ねて94.6%。

**予算を上げる側**: 幅(`--gates 8000`)で94.6→**97.1%**(3シード平均、約86kゲート)、ただしその先に**暗記の壁**(train=100%でtestが97.0%に凍結)。学習済み3ネットの投票で**97.6%**(このリポジトリのMNIST最高値、約200kゲート・推論3回)。ただし**ゲート効率では勾配学習の先行研究に明確に負けています**(difflogic 48k/97.69%、LILogic 8k/98.45%)— 隠さず正直に書いています。

全実験(各レバーの勝ち負け・反証された仮説・診断・逆伝播の正確な会計)は **[RESULTS.md](docs/RESULTS.md)**(スケーリング系は[SCALING.md](docs/SCALING.md))に、生ログは実験ごとのissueにあります。新しい実験はそちらに追記します。構成要素のほとんどは先行研究からの借り物で、新規性は主張しません(詳細は英語本文「What this borrows」)。
