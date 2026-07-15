"""
greedy-lgn: Backprop-free, layer-by-layer training of Logic Gate Networks
with immediate discretization, adaptive depth, and incremental logic simplification.
CLI entry point; the implementation lives in core / groupsum / ff / greedy /
scaling / e2e / simplify modules.

Runs on CPU in a few minutes. Requirements: torch, scikit-learn.

Usage:
    python experiment.py                     # default config (a few minutes on CPU)
    python experiment.py --gates 200 --epochs 30 --max-layers 3   # quick smoke test
    python experiment.py --device cuda       # same experiment on GPU
    python experiment.py --skip-input        # re-expose input bits to every layer
    python experiment.py --skip-all          # DenseNet-style: all previous layers
    python experiment.py --window 4 --commit 1   # receding horizon: look 4 ahead, commit 1
    python experiment.py --ensemble 4            # 4 independent nets + voting
    python experiment.py --objective ff          # Forward-Forward local objective
    python experiment.py --objective ff --ff-struct 0.5 --ff-label-rep 1   # structured data x label wiring
    python experiment.py --dataset mnist --device cuda --batch 4096 --epochs 30   # MNIST


Regression: any change to these modules must keep every number in tests.py
exact (`python tests.py`). The published results depend on bit-exact
reproducibility.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))  # 実装はsrc/
import numpy as np
import torch
from core import load_data, reps
from ff import ff_inputs
from greedy import run_greedy
from scaling import run_ensemble
from e2e import run_e2e
from simplify import simplify

# ----------------------------- main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gates", type=int, default=500, help="gates per layer (multiple of 10)")
    p.add_argument("--epochs", type=int, default=120, help="epochs per greedy layer"
                   " (upper bound when --epoch-stop is set)")
    p.add_argument("--epoch-stop", type=float, default=0.0, metavar="T",
                   help="adaptive epochs: stop a layer early once its gate-argmax"
                        " (the discrete circuit) change-rate stays below T for"
                        " --epoch-patience checks. 0 = off (fixed --epochs)."
                        " --epochs is the upper bound. groupsum only.")
    p.add_argument("--epoch-peak", type=float, default=0.0, metavar="F",
                   help="weak-learner mode: stop a layer once its argmax change"
                        "-rate decays below F x the peak rate seen so far (commit"
                        " half-baked layers early, let depth do the work). 0 = off."
                        " Overrides --epoch-stop. Try F=0.5 with --epoch-min 20.")
    p.add_argument("--epoch-peak-decay", type=float, default=1.0, metavar="D",
                   help="depth schedule for --epoch-peak: layer d uses F*D^d, so"
                        " early layers fold fast (weak learners) and deeper layers"
                        " train ever closer to saturation. 1.0 = constant F.")
    p.add_argument("--epoch-chain", type=float, default=0.0, metavar="M",
                   help="chained anchor: layer 1 settles via --epoch-stop and its"
                        " stop-time churn rate becomes the yardstick; each later"
                        " layer stops when its churn decays below M x the previous"
                        " layer's stop rate (auto-calibrated threshold; M=1 same"
                        " neighbourhood, M>1 fold earlier each generation). 0 = off."
                        " Needs --epoch-stop for the first layer.")
    p.add_argument("--epoch-min", type=int, default=70, metavar="M",
                   help="do not stop before epoch M (protects warm-start identity"
                        " layers: their churn dips in a quiet valley around epoch"
                        " 30-60 before ramping up, and min=30 false-fired there --"
                        " calibrated on digits, layers settle naturally at 140-190)")
    p.add_argument("--epoch-check", type=int, default=5, metavar="K",
                   help="check the argmax change-rate every K epochs")
    p.add_argument("--epoch-patience", type=int, default=3, metavar="P",
                   help="consecutive sub-threshold checks required to stop a layer")
    p.add_argument("--max-layers", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--e2e-max-epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-e2e", action="store_true", help="skip the backprop baseline")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--e2e-depth", type=int, default=None,
                   help="override e2e baseline depth (default: greedy's chosen depth)")
    p.add_argument("--skip-input", action="store_true",
                   help="concatenate the input bits into every greedy layer's wiring"
                        " pool (skip connections; e2e baseline is unaffected)")
    p.add_argument("--skip-all", action="store_true",
                   help="DenseNet-style: wiring pool = input bits + ALL previous"
                        " layers' outputs (overrides --skip-input)")
    p.add_argument("--window", type=int, default=1,
                   help="receding-horizon lookahead: jointly train WINDOW fresh soft"
                        " layers with backprop on top of the frozen prefix"
                        " (1 = plain greedy, the original behaviour)")
    p.add_argument("--commit", type=int, default=1,
                   help="layers discretized+frozen per window slide (1..WINDOW;"
                        " commit=window = non-overlapping block greedy)")
    p.add_argument("--carry", action="store_true",
                   help="growing-scaffold window: instead of discarding the"
                        " uncommitted lookahead layers each slide, carry their"
                        " trained weights into the next window (warm-start). Lets"
                        " layers keep growing across slides. no-skip only")
    p.add_argument("--win-loss", choices=["last", "all"], default="last",
                   help="window training loss: CE at the last window layer only"
                        " (pure lookahead) or averaged over all window layers"
                        " (deep supervision). groupsum only; ff always uses all")
    p.add_argument("--group-loss", choices=["ce", "bce"], default="ce",
                   help="groupsum local loss: ce (cross-entropy on the scaled group"
                        " sums, original) or bce (per-class BCE: correct class's bit"
                        " group -> 1, others -> 0, on group means in [0,1])")
    p.add_argument("--group-residual", action="store_true",
                   help="boosting readout: each layer's class scores are added to the"
                        " frozen layers' accumulated prediction (each layer learns to"
                        " correct the running residual). Prediction = argmax of the"
                        " total class-c bits over all layers. groupsum, ce, window=1")
    p.add_argument("--group-boost", type=float, default=1.0, metavar="B",
                   help="AdaBoost-style sample reweighting on top of residual:"
                        " samples the frozen running sum currently misclassifies get"
                        " their CE weighted B times more in the next layer's training"
                        " (1.0 = off, uniform). Needs --group-residual.")
    p.add_argument("--seq", action="store_true",
                   help="row-sequential recurrence (RDDLGN-style temporal state):"
                        " present the image one row per step; each greedy layer is"
                        " a recurrent cell s_t = L([x_t; s_(t-1)]) trained by BPTT"
                        " over T steps, then discretized so frozen layers pass HARD"
                        " state bits to the next layer. Readout = GroupSum of the"
                        " final state. groupsum + window=1 + no-skip only.")
    p.add_argument("--recur", type=int, default=1, metavar="K",
                   help="within-layer recursion: apply each layer K times with"
                        " shared weights (wiring must line up, so layer 1 -- input"
                        " width -- runs once and layers 2+ iterate). Learned logits"
                        " stay 1 layer's worth; the unrolled circuit is K layers"
                        " deep per trained layer. no-skip + groupsum only.")
    p.add_argument("--warm-start", type=float, default=0.0, metavar="B",
                   help="identity init: each new layer (that has a previous layer)"
                        " starts by reproducing the previous layer's output bits"
                        " (gate-A passthrough, logit biased by B toward A) instead of"
                        " random, then learns the residual from there (ResNet-style"
                        " identity block). 0 = off, random init. groupsum only.")
    p.add_argument("--objective", choices=["groupsum", "ff"], default="groupsum",
                   help="per-layer local objective: groupsum (GroupSum+CE, original)"
                        " or ff (Forward-Forward goodness = popcount on binary"
                        " layers; labels are overlaid on the input, inference tries"
                        " all 10 labels)")
    p.add_argument("--ff-neg", choices=["random", "hard", "mix", "review"],
                   default="random",
                   help="negative-label policy for ff: random wrong label (original),"
                        " hard (the most plausible wrong label, re-mined per layer),"
                        " mix (hard/random 50/50), or review (misclassified samples"
                        " study their own wrong answer, correct ones stay random)")
    p.add_argument("--ff-neg-warmup", type=float, default=0.0,
                   help="fraction of each layer's epochs trained with random"
                        " negatives before mining (0 = mine from the frozen prefix"
                        " only, from layer 2 on; >0 = the partially trained layer"
                        " itself takes the mock exam, so layer 1 participates too)")
    p.add_argument("--ff-neg-phases", type=int, default=1,
                   help="split the post-warmup epochs into this many phases,"
                        " re-taking the mock exam and re-mining negatives before"
                        " each (1 = mine once; >1 needs --ff-neg-warmup > 0 to"
                        " differ, since the frozen-prefix grader is static)")
    p.add_argument("--ff-neg-boost", type=float, default=1.0,
                   help="weight the loss of currently-misclassified samples this"
                        " many times higher (spend more gradient on hard examples;"
                        " 1.0 = uniform, the original behaviour)")
    p.add_argument("--ff-label-rep", type=int, default=1,
                   help="replicate the 10 overlaid label bits this many times so"
                        " random wiring actually samples them (ff objective only)")
    p.add_argument("--ff-struct", type=float, default=0.0,
                   help="fraction of the FIRST layer's gates forced to wire"
                        " data x label (one input from the data bits, one from the"
                        " label bits) instead of relying on label replication; kills"
                        " the wasted label x label gates. 0 = off. ff objective only;"
                        " works with --ff-label-rep 1 since access is guaranteed")
    p.add_argument("--ensemble", type=int, default=1,
                   help="train ENSEMBLE independent greedy networks (seeds seed.."
                        "seed+M-1) side by side and report soft-vote / majority-vote"
                        " accuracy (1 = single network, the original behaviour)")
    p.add_argument("--dataset", choices=["digits", "mnist", "cifar10"],
                   default="digits",
                   help="digits: sklearn 8x8 (CPU-friendly). mnist: 28x28, 70k"
                        " samples (GPU + --batch recommended). cifar10: 32x32x3,"
                        " 60k samples, per-channel thermometer -> 9 input planes"
                        " (GPU + --batch; first run downloads ~170MB from"
                        " cs.toronto.edu, md5-verified)")
    p.add_argument("--conv", type=int, default=0, metavar="C",
                   help="convolutional logic layers, phase 2 (weight-shared"
                        " kernels + OR-pooling, after conv-DLGN Petersen et al."
                        " 2024): C channels, each a depth-TREE binary tree of"
                        " gates whose leaves wire randomly inside a KxK window,"
                        " replicated over all positions, then POOLxPOOL"
                        " max(=OR)-pooled. Learned logits per layer ="
                        " C*(2^TREE-1) gates; unrolled circuit = HxW copies."
                        " 0 = off. groupsum + window=1 + no-skip only.")
    p.add_argument("--conv-k", type=int, default=3, metavar="K",
                   help="conv kernel window size")
    p.add_argument("--conv-tree", type=int, default=2, metavar="T",
                   help="gate-tree depth per kernel (2^T leaves, 2^T-1 gates)")
    p.add_argument("--conv-pool", type=int, default=2, metavar="P",
                   help="pooling factor per conv layer (1 = no pooling;"
                        " automatically disabled once the map is too small)")
    p.add_argument("--conv-sched", type=str, default=None, metavar="C0,C1,...",
                   help="per-layer channel schedule for --conv (overrides the flat"
                        " --conv count): e.g. 128,64,32 = inverted-funnel (wide"
                        " first, after V1's LGN->V1 fan-out). Depths past the list"
                        " reuse the last value. Sets --conv to the first entry.")
    p.add_argument("--local", type=int, default=0, metavar="K",
                   help="convolutional wiring, phase 1 (locality prior only, no"
                        " weight sharing): every gate gets a pixel position and"
                        " draws its two inputs from the K x K neighbourhood of the"
                        " pool (inputs at their pixel, previous gates at their"
                        " assigned position; gate positions are random to stay"
                        " uncorrelated with the GroupSum class groups, or inherited"
                        " from the previous layer under --warm-start). 0 = off,"
                        " original global random wiring. groupsum + window=1 only;"
                        " --skip-input supported, --skip-all/--recur/--seq not.")
    p.add_argument("--thresholds", type=str, default=None, metavar="SPEC",
                   help="input binarization override: \"5,10,15\" = absolute"
                        " thermometer thresholds, \"q4\" = 4 planes at evenly"
                        " spaced quantiles of the nonzero TRAIN pixels (also the"
                        " way to add planes). Default: the original fixed"
                        " (3,7,11) digits / (63,127,191) mnist+cifar10 (per"
                        " channel), bit-identical.")
    p.add_argument("--batch", type=int, default=0,
                   help="minibatch size (0 = full batch, the original behaviour;"
                        " required in practice for mnist on a 6 GB GPU)")
    cfg = p.parse_args()
    if not (1 <= cfg.commit <= cfg.window):
        p.error("--commit must satisfy 1 <= commit <= window")
    if cfg.carry and (cfg.skip_input or cfg.skip_all):
        p.error("--carry is no-skip only (carried layers assume constant in_dim)")
    if cfg.group_residual and (cfg.objective != "groupsum" or cfg.group_loss != "ce"):
        p.error("--group-residual needs groupsum objective and ce loss")
    if cfg.group_boost != 1.0 and not cfg.group_residual:
        p.error("--group-boost needs --group-residual")
    if cfg.warm_start > 0 and cfg.objective != "groupsum":
        p.error("--warm-start needs groupsum objective")
    if (cfg.epoch_stop > 0 or cfg.epoch_peak > 0) and cfg.objective != "groupsum":
        p.error("--epoch-stop/--epoch-peak need groupsum objective")
    if cfg.epoch_chain > 0 and (cfg.epoch_stop <= 0 or cfg.epoch_peak > 0):
        p.error("--epoch-chain needs --epoch-stop (first-layer criterion)"
                " and is exclusive with --epoch-peak")
    if cfg.recur > 1 and (cfg.skip_input or cfg.skip_all
                          or cfg.objective != "groupsum" or cfg.carry
                          or cfg.group_residual):
        p.error("--recur is no-skip + groupsum only (iteration needs the pool"
                " width to equal --gates), and exclusive with --group-residual"
                " (readout semantics of intermediate iterates is unresolved)")
    if cfg.seq and (cfg.objective != "groupsum" or cfg.window > 1
                    or cfg.skip_input or cfg.skip_all or cfg.carry
                    or cfg.recur > 1 or cfg.group_residual):
        p.error("--seq is groupsum + window=1 + no-skip only (no recur/residual"
                " combination yet)")
    if cfg.local > 0 and (cfg.objective != "groupsum" or cfg.window > 1
                          or cfg.skip_all or cfg.carry or cfg.recur > 1
                          or cfg.seq):
        p.error("--local is groupsum + window=1 only (skip-input OK;"
                " skip-all/carry/recur/seq not yet)")
    # --conv-sched "128,64,32" を数値リスト化し、cfg.conv を先頭に(オフ判定用)
    cfg.conv_sched = ([int(x) for x in cfg.conv_sched.split(",")]
                      if cfg.conv_sched else None)
    if cfg.conv_sched:
        cfg.conv = cfg.conv_sched[0]
    if cfg.conv > 0 and (cfg.objective != "groupsum" or cfg.window > 1
                         or cfg.skip_input or cfg.skip_all or cfg.carry
                         or cfg.recur > 1 or cfg.seq or cfg.local > 0
                         or cfg.warm_start > 0):
        p.error("--conv is groupsum + window=1 + no-skip only (no recur/seq/"
                "local/warm-start combination yet)")
    cfg.n_class = 10
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in
                          load_data(cfg.dataset, thresholds=cfg.thresholds)]
    print(f"data: {cfg.dataset}, {Xtr.shape[0]} train / {Xte.shape[0]} test,"
          f" {Xtr.shape[1]} input bits  (device={cfg.device}"
          + (f", batch={cfg.batch}" if cfg.batch else "") + ")\n")

    # (FFの簡略化検証: 入力はラベル重畳込み。簡略化器が表示するaccはGroupSum
    #  読み出しなのでFFの精度ではないが、ビット等価性の検証はそのまま有効)
    Xte_s = (ff_inputs(Xte, yte, cfg.n_class, cfg.ff_label_rep)
             if cfg.objective == "ff" else Xte)

    def unroll(ls):
        """--recur>1: 簡略化・検証には反復を展開した回路(同じ層のK回並び)を渡す。
        反復回数は実行時と同じ規則(プール幅==gatesのときK回)。recur=1では ls のまま"""
        out, w = [], Xte_s.shape[1]
        for L in ls:
            out += [L.cpu()] * reps(w, cfg)
            w = cfg.gates
        return out

    if cfg.ensemble > 1:
        members, member_acc, depths, soft_acc, maj_acc = run_ensemble(
            Xtr, Xte, ytr, yte, cfg)
        e2e_soft = e2e_hard = None
        if not cfg.skip_e2e:
            e2e_soft, e2e_hard = run_e2e(Xtr, Xte, ytr, yte,
                                         cfg.e2e_depth or depths[0], cfg)
        before = after = 0
        if cfg.seq or cfg.conv:
            print("=== (C) simplification skipped (seq/conv) ===\n")
        else:
            for ls in members:  # メンバーごとに簡略化+ビット等価検証
                b, a = simplify(unroll(ls), Xte_s.cpu(), yte.cpu(), cfg)
                before += b; after += a
        summary = {"member_hard_test_acc": [round(a, 4) for a in member_acc],
                   "member_mean": round(float(np.mean(member_acc)), 4),
                   "ensemble_soft_vote_acc": round(soft_acc, 4),
                   "ensemble_majority_vote_acc": round(maj_acc, 4),
                   "depths": depths,
                   "e2e_soft_test_acc": e2e_soft and round(e2e_soft, 4),
                   "e2e_hard_test_acc": e2e_hard and round(e2e_hard, 4),
                   "gates_before": before, "gates_after_simplify": after}
        print("=== summary ===")
        print(json.dumps(summary, indent=2))
        return

    layers, greedy_acc, depth = run_greedy(Xtr, Xte, ytr, yte, cfg)
    e2e_soft = e2e_hard = None
    if not cfg.skip_e2e:
        e2e_soft, e2e_hard = run_e2e(Xtr, Xte, ytr, yte, cfg.e2e_depth or depth, cfg)
    # simplification is pure-Python graph rewriting -> always run on CPU.
    # residual: simplifyが全層を出力扱いにするので検証込みで実行できる(2026-07-13)
    if cfg.seq:
        # 時系列回路はレジスタ(状態)を含み、現簡略化器(組合せ回路前提)の対象外。
        # 時間展開しての検証は将来課題
        print("=== (C) simplification skipped (seq: sequential circuit with"
              " registers) ===\n")
        before = after = 0
    elif cfg.conv:
        # 位置展開(H*W*C*木)でのゲート列挙は可能だが未実装(将来課題)
        print("=== (C) simplification skipped (conv: position-unrolled"
              " enumeration not implemented yet) ===\n")
        before = after = 0
    else:
        before, after = simplify(unroll(layers), Xte_s.cpu(), yte.cpu(), cfg)

    summary = {"objective": cfg.objective,
               "greedy_hard_test_acc": round(greedy_acc, 4),
               "greedy_depth": depth,
               "e2e_soft_test_acc": e2e_soft and round(e2e_soft, 4),
               "e2e_hard_test_acc": e2e_hard and round(e2e_hard, 4),
               "gates_before": before, "gates_after_simplify": after}
    print("=== summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
