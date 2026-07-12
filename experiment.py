"""
greedy-lgn: Backprop-free, layer-by-layer training of Logic Gate Networks
with immediate discretization, adaptive depth, and incremental logic simplification.

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

Regression: any change to this file must keep every number in tests.py exact
(`python tests.py`). The published results depend on bit-exact reproducibility.
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ----------------------------- 16 two-input gates -----------------------------
# Real-valued relaxations (exact on {0,1}). Order follows difflogic convention.
GATE_NAMES = ["FALSE", "AND", "A&!B", "A", "!A&B", "B", "XOR", "OR",
              "NOR", "XNOR", "!B", "A|!B", "!A", "!A|B", "NAND", "TRUE"]
SWAP = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6, 7: 7, 8: 8, 9: 9,
        10: 12, 11: 13, 12: 10, 13: 11, 14: 14, 15: 15}  # fn under input swap

def all16(a, b):
    ab = a * b
    return torch.stack([
        torch.zeros_like(a), ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
        1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a,
        1 - a + ab, 1 - ab, torch.ones_like(a)], dim=-1)

GATE_A = 3   # all16のindex 3 = 第1入力aのパススルー(恒等warm-startで使用)

def f16(fn, a, b):
    ab = a * b
    return [a * 0, ab, a - ab, a, b - ab, b, a + b - 2 * ab, a + b - ab,
            1 - (a + b - ab), 1 - (a + b - 2 * ab), 1 - b, 1 - b + ab, 1 - a,
            1 - a + ab, 1 - ab, a * 0 + 1][fn]

class LogicLayer(nn.Module):
    """One layer of 2-input logic gates with fixed random wiring.

    struct=(data_hi, lab_lo, lab_hi, frac, seed) で「割合fracのゲートを
    ia=データ範囲[0,data_hi) / ib=ラベル範囲[lab_lo,lab_hi) に強制配線」する
    構造化モード(FFのラベル×ラベル無駄ゲート対策)。主ドロー(ia/ib/logits)を
    従来順で引いた後に別ジェネレータで一部を上書きするので、struct=None または
    frac=0 のとき従来とビット単位で一致する。"""
    def __init__(self, in_dim, n_gates, seed, struct=None, warm=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        ia = torch.randint(0, in_dim, (n_gates,), generator=g)
        ib = torch.randint(0, in_dim, (n_gates,), generator=g)
        logits = torch.randn(n_gates, 16, generator=g)
        # warm>0: 恒等初期化。前層出力h(プール末尾n_gates本 — no-skip/skip共通)を
        # gate A のパススルーで再現し、logitをAに強さwarmだけ偏らせる。新層は前層を
        # 壊さない状態から残差だけ微調整して始まる(ResNetの恒等ブロック初期化)。
        # in_dim>=n_gates が必要(前層があれば成立)。warm=0で従来とビット一致
        if warm > 0 and in_dim >= n_gates:
            ia = torch.arange(n_gates) + (in_dim - n_gates)  # gate i <- 前層ビットi
            logits = logits.clone()
            logits[:, GATE_A] += warm
        if struct is not None:
            data_hi, lab_lo, lab_hi, frac, sseed = struct
            k = int(round(n_gates * frac))
            if k > 0 and data_hi > 0 and lab_hi > lab_lo:
                gs = torch.Generator().manual_seed(sseed)   # 主ドローと独立
                pos = torch.randperm(n_gates, generator=gs)[:k]  # 構造化するゲート
                ia[pos] = torch.randint(0, data_hi, (k,), generator=gs)
                ib[pos] = torch.randint(lab_lo, lab_hi, (k,), generator=gs)
        self.register_buffer("ia", ia)
        self.register_buffer("ib", ib)
        self.logits = nn.Parameter(logits)
    def forward(self, x):  # soft (training) mode
        return (all16(x[:, self.ia], x[:, self.ib])
                * F.softmax(self.logits, -1)).sum(-1)
    @torch.no_grad()
    def hard(self, x):  # discretized (inference) mode
        sel = self.logits.argmax(-1)
        return all16(x[:, self.ia], x[:, self.ib]).gather(
            -1, sel.view(1, -1, 1).expand(x.shape[0], -1, 1)).squeeze(-1)

# ----------------------------- data -----------------------------
def load_data(dataset="digits", seed=0):
    if dataset == "mnist":
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="liac-arff")  # no pandas needed
        y = y.astype(np.int64)                   # 28x28 pixels, values 0..255
        Xb = np.concatenate([(X > t).astype(np.float32) for t in (63, 127, 191)], axis=1)
        return (torch.tensor(Xb[:60000]), torch.tensor(Xb[60000:]),   # standard split
                torch.tensor(y[:60000]), torch.tensor(y[60000:]))
    X, y = load_digits(return_X_y=True)          # 8x8 digits, values 0..16
    Xb = np.concatenate([(X > t).astype(np.float32) for t in (3, 7, 11)], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(
        Xb, y, test_size=0.25, stratify=y, random_state=seed)
    return (torch.tensor(Xtr), torch.tensor(Xte),
            torch.tensor(ytr), torch.tensor(yte))

# ----------------------------- training utils -----------------------------
def _group_sizes(D, n_class):
    # D本の出力ビットをn_class群に分割。割り切れない端数r本は先頭r群に1本ずつ配る
    return [D // n_class + (1 if i < D % n_class else 0) for i in range(n_class)]

def group_sum(h, n_class, tau):
    if h.shape[1] % n_class == 0:                       # 割り切れる=現行の高速パス
        return h.view(h.shape[0], n_class, -1).sum(-1) / tau   # (ビット単位で従来一致)
    return torch.stack([c.sum(1) for c in            # 端数あり: split で群ごと集計
                        h.split(_group_sizes(h.shape[1], n_class), 1)], 1) / tau

def group_mean(h, n_class):
    # クラス別スコアを平均で集計(各ビット∈[0,1]なので mean∈[0,1]、BCE目標0/1向き)。
    # argmaxはsumと同順なのでプローブ・推論には影響せず、BCEロス専用
    if h.shape[1] % n_class == 0:
        return h.view(h.shape[0], n_class, -1).mean(-1)
    return torch.stack([c.mean(1) for c in
                        h.split(_group_sizes(h.shape[1], n_class), 1)], 1)

def accuracy(logits, y):
    return (logits.argmax(-1) == y).float().mean().item()

def next_pool(h, X, pool_prev, cfg):
    """あるレイヤーの出力hを受けて、その次のレイヤーの配線プールを作る
    (soft/hard共通。凍結時と窓内softフォワードで同一規則を使う)"""
    if cfg.skip_all:      # DenseNet-style: input + every previous layer
        return torch.cat([pool_prev, h], 1)
    if cfg.skip_input:    # input + previous layer only
        return torch.cat([X, h], 1)
    return h

@torch.no_grad()
def hard_batched(layer, x, budget=8192 * 500):  # bound the [B, G, 16] temporary
    chunk = max(1024, budget // layer.ia.numel())  # 500 gates -> 8192 rows (as before)
    return layer.hard(x) if len(x) <= chunk else torch.cat(
        [layer.hard(c) for c in x.split(chunk)])

def hard_pass(layers, X, cfg):
    """凍結済み層をhardで順に評価し、(最終層の出力h, 次層の配線プール)を返す。
    layersが空のときは (None, X)"""
    pool, h = X, None
    for L in layers:
        h = hard_batched(L, pool)
        pool = next_pool(h, X, pool, cfg)
    return h, pool

def fit(loss, n, cfg, seed, epochs, opt, stop_check=None):
    """full-batch / minibatch 共通のエポックループ。loss(idx) は idx=None で
    全バッチ、Tensorでその行だけの損失を返すクロージャ。
    stop_check(epoch)->bool を渡すと各エポック後に呼び、Trueで早期終了
    (適応エポック=頭打ち検出)。stop_check=None のとき従来と完全一致(ビット等価)"""
    if not cfg.batch or cfg.batch >= n:   # full-batch (default)
        for e in range(1, epochs + 1):
            opt.zero_grad()
            loss(None).backward()
            opt.step()
            if stop_check is not None and stop_check(e):
                break
        return
    g = torch.Generator().manual_seed(seed)
    for e in range(1, epochs + 1):
        for idx in torch.randperm(n, generator=g).split(cfg.batch):
            idx = idx.to(cfg.device)
            opt.zero_grad()
            loss(idx).backward()
            opt.step()
        if stop_check is not None and stop_check(e):
            break

def make_stop_check(win, cfg, d0=0, obj=None):
    """適応エポックの停止判定を作る。win内の全ゲートのargmax選択(=離散回路)を
    epoch_checkごとにスナップショットし、変化率(churn)で層を早期終了する。
    2モード:
      --epoch-stop T : 飽和基準。churn < T が epoch_patience回連続で停止
      --epoch-peak F : 弱学習器基準。churnが観測ピークのF倍未満に減衰したら停止
        (飽和を待たず「学習の山を越えたら」畳む。半煮えの層を量産して深さに
        仕事をさせるブースティング的発想)
    --epoch-peak-decay D は深さスケジュール: 層d(凍結済み層数)の実効F=F*D^d。
    浅い層は早畳み・深い層ほど飽和まで粘る(F→0でchurn完全停止まで待つのと同等。
    総深さは自動選択で事前に不明なので、地平線不要の指数減衰にしている)。
    --epoch-chain M は連鎖アンカー: 最初の層は飽和基準(--epoch-stop)で止め、
    その発火時のchurn率を「収束の物差し」として記録。以降の各層は自分のchurnが
    M×(前層の停止時churn率)未満に減衰したら畳む(閾値の手調整を層1が自動校正。
    M=1で前層と同じ付近、M>1で世代ごとに早畳み)。
    epoch_min前は判定しない(warm恒等層は序盤ほぼ変化ゼロ→活発化→減衰と
    遅れて立ち上がるため、学習前の静かな谷での誤発火を防ぐ)。
    どれも0のときは None を返し、従来の固定エポックのまま(ビット等価)"""
    if cfg.epoch_stop <= 0 and cfg.epoch_peak <= 0:
        return None
    thr = cfg.epoch_stop
    if cfg.epoch_chain > 0 and d0 > 0 and getattr(obj, "stop_rate", None) is not None:
        # 前層の停止時churnをアンカーに(床=1ゲート分。rate量子=1/gates未満は無意味)
        thr = max(cfg.epoch_chain * obj.stop_rate, 1.0 / cfg.gates)
    f_eff = cfg.epoch_peak * (cfg.epoch_peak_decay ** d0)
    state = {"prev": None, "hits": 0, "peak": 0.0}
    @torch.no_grad()
    def check(e):
        if e < cfg.epoch_min or e % cfg.epoch_check != 0:
            return False
        sel = torch.cat([L.logits.argmax(-1) for L in win])
        prev = state["prev"]
        state["prev"] = sel
        if prev is None:
            return False
        rate = (sel != prev).float().mean().item()
        if cfg.epoch_peak > 0:
            state["peak"] = max(state["peak"], rate)
            fired = state["peak"] > 0 and rate < f_eff * state["peak"]
        else:
            fired = rate < thr
        state["hits"] = state["hits"] + 1 if fired else 0
        if state["hits"] >= cfg.epoch_patience:
            if cfg.epoch_chain > 0 and obj is not None:
                obj.stop_rate = rate       # 次層のアンカーとして記録
            print(f"    (epoch-stop: layer done at epoch {e}, rate={rate:.4f}"
                  + (f", peak={state['peak']:.4f} F={f_eff:.3f}"
                     if cfg.epoch_peak > 0 else
                     f", thr={thr:.4f}" if cfg.epoch_chain > 0 else "")
                  + ")")
            return True
        return False
    return check

# ----------------------------- local objectives -----------------------------
# greedyループ(run_greedy)は目的関数に依存しない骨格で、各objectiveが
# begin(窓の準備)/train(窓の学習)/commit(離散化後のプローブ)/counts(投票用
# クラス別スコア)を提供する。obj.X は配線プールの基底(skip-input用)。

class GroupSum:
    """従来のローカル目的: 各層のGroupSum+CE readout。プールは増分更新で保持"""
    kind, tag = "hard", "greedy"
    def __init__(self, Xtr, Xte, ytr, yte, cfg):
        self.cfg, self.ytr, self.yte = cfg, ytr, yte
        self.X, self.Xte = Xtr, Xte              # 配線プールの基底(skip用)
        self.pool_tr, self.pool_te = Xtr, Xte
        self.tau = float(np.sqrt(cfg.gates / cfg.n_class))
        # --group-residual: 凍結層のクラススコアを累積し、各層は「前層までの累積
        # 予測」を固定オフセットに残差を埋めるよう学習(ブースティング)。累積は
        # 全層のクラス別ビット総和 = 純論理回路のまま
        if cfg.group_residual:
            self.accum_tr = torch.zeros(len(ytr), cfg.n_class, device=cfg.device)
            self.accum_te = torch.zeros(len(yte), cfg.n_class, device=cfg.device)
    def header(self):
        cfg = self.cfg
        return ("=== (A) Greedy layer-wise: local loss -> discretize -> freeze ==="
                + (" [residual]" if cfg.group_residual else "")
                + (f" [boost={cfg.group_boost}]" if cfg.group_boost != 1.0 else "")
                + (f" [warm-start={cfg.warm_start}]" if cfg.warm_start > 0 else "")
                + (f" [epoch-peak={cfg.epoch_peak}"
                   + (f" decay={cfg.epoch_peak_decay}"
                      if cfg.epoch_peak_decay != 1.0 else "")
                   + f" min={cfg.epoch_min} cap={cfg.epochs}]"
                   if cfg.epoch_peak > 0 else
                   f" [epoch-stop={cfg.epoch_stop}"
                   + (f" chain={cfg.epoch_chain}" if cfg.epoch_chain > 0 else "")
                   + f" min={cfg.epoch_min} cap={cfg.epochs}]"
                   if cfg.epoch_stop > 0 else "")
                + (f" [loss={cfg.group_loss}]" if cfg.group_loss != "ce" else "")
                + (f" [window={cfg.window} commit={cfg.commit} loss={cfg.win_loss}]"
                   if cfg.window > 1 else ""))
    def begin(self, layers, d0):   # 窓学習の準備。窓の入力次元を返す
        return self.pool_tr.shape[1]
    def make_layer(self, in_dim, d0, k):
        # 前層があるとき(d0+k>0)だけ恒等warm-start。最初の層は前層が無いので素のまま
        warm = self.cfg.warm_start if (d0 + k) > 0 else 0.0
        return LogicLayer(in_dim, self.cfg.gates,
                          seed=self.cfg.seed * 100 + d0 + k + 1, warm=warm)
    def train(self, win, layers, d0):
        """窓内W層をsoftのまま共同学習(receding horizonの1ステップ)。
        損失は窓の最終層のCE(--win-loss last)か全層平均(--win-loss all)。
        W=1のとき従来のgreedy 1層学習と厳密に一致する"""
        cfg = self.cfg
        opt = torch.optim.Adam([p for L in win for p in L.parameters()], lr=cfg.lr)
        def loss(idx):
            pool = self.pool_tr if idx is None else self.pool_tr[idx]
            X = self.X if idx is None else self.X[idx]
            y = self.ytr if idx is None else self.ytr[idx]
            # residual: 凍結prefixの累積accumを引き継ぎ、窓内の各層の寄与を足して
            # いく(boostingをwindowに拡張=①+②の土台)。非residualはrun=None
            run = (None if not cfg.group_residual else
                   self.accum_tr if idx is None else self.accum_tr[idx])
            # --group-boost: 凍結累積(=run初期値)が誤答のサンプルのCEをB倍に
            # 傾斜(AdaBoost式のサンプル再重み付け)。採点者は凍結prefixのみ
            # (窓内の未凍結寄与を含めない)。layers空(最初の層)は累積ゼロで
            # 誤答判定が縮退するので一律。B=1.0は従来と厳密一致(回帰維持)
            wvec = None
            if cfg.group_boost != 1.0 and run is not None and layers:
                wvec = torch.where(run.argmax(1) != y,
                                   float(cfg.group_boost), 1.0)
            def term(h):
                # ce=scaled-sum(residualは累積run)のCE、bce=group-meanのBCE
                if cfg.group_loss == "bce":
                    return F.binary_cross_entropy(group_mean(h, cfg.n_class),
                                                  F.one_hot(y, cfg.n_class).float())
                logits = run if cfg.group_residual else group_sum(h, cfg.n_class,
                                                                  self.tau)
                if wvec is None:
                    return F.cross_entropy(logits, y)
                ce = F.cross_entropy(logits, y, reduction="none")
                return (ce * wvec).sum() / wvec.sum()
            h, terms = None, []
            for L in win:
                h = L(pool)
                if cfg.group_residual:
                    g = group_sum(h, cfg.n_class, self.tau)
                    run = g if run is None else run + g
                if cfg.win_loss == "all":
                    terms.append(term(h))
                pool = next_pool(h, X, pool, cfg)
            return (sum(terms) / len(terms) if cfg.win_loss == "all" else term(h))
        fit(loss, len(self.pool_tr), cfg, cfg.seed * 1000 + d0 + 1, cfg.epochs, opt,
            stop_check=make_stop_check(win, cfg, d0, self))
    def commit(self, layers, L):
        """Lを離散化・凍結してプールをHARDビットで前進。(train, test)プローブを返す。
        residual時は累積スコアに当層の寄与を足し、プローブは累積で測る"""
        cfg = self.cfg
        h_tr, h_te = hard_batched(L, self.pool_tr), hard_batched(L, self.pool_te)
        s_tr = group_sum(h_tr, cfg.n_class, self.tau)
        s_te = group_sum(h_te, cfg.n_class, self.tau)
        if cfg.group_residual:
            self.accum_tr = self.accum_tr + s_tr    # 累積更新(= 次層のオフセット)
            self.accum_te = self.accum_te + s_te
            s_tr, s_te = self.accum_tr, self.accum_te
        a_te = accuracy(s_te, self.yte)
        a_tr = accuracy(s_tr, self.ytr)
        self.pool_tr = next_pool(h_tr, self.X, self.pool_tr, cfg)
        self.pool_te = next_pool(h_te, self.Xte, self.pool_te, cfg)
        return a_tr, a_te
    @torch.no_grad()
    def counts(self, layers, X, y):
        """アンサンブル投票用のクラス別スコア [B, n_class](=GroupSumカウント)。
        τで割らず厳密な整数カウントを返す: argmaxには数学的に無関係だが、
        τ除算後のfloatをメンバー間で合算すると、合計が同点のクラス同士で
        丸め順序の差によりargmaxがCPU/GPUで割れる(tests.pyが発見した
        soft voteの不一致 0.9133 vs 0.9111 の原因と修正)"""
        h, _ = hard_pass(layers, X, self.cfg)
        return group_sum(h, self.cfg.n_class, 1.0)

def ff_inputs(X, y, n_class, rep=1):
    """FF流にラベルを入力へ重畳: one-hot 10ビットをrep回複製して連結する。
    ランダム2入力配線では、ラベルビットがプールの一定割合を占めないと
    ほとんどのゲートがラベルを見ない(rep=1だと約10%しか触れない)"""
    return torch.cat([X, F.one_hot(y, n_class).float().repeat(1, rep)], 1)

class ForwardForward:
    """FF目的: goodness(バイナリ層ではpopcount)を正例で上げ負例で下げる。
    ラベルは入力に重畳するので層にreadoutは無く、推論は全ラベル試行。
    層内は勾配学習のまま(FFは目的関数の置換であり勾配フリー化ではない)。
    窓損失は常に全層平均(deep supervision) — コミット層のgoodnessが深さ選択の
    プローブかつ推論のreadoutなので、groupsumの--win-loss last相当は成立しない"""
    kind, tag = "goodness", "greedy-FF"
    def __init__(self, Xtr, Xte, ytr, yte, cfg):
        self.cfg, self.Xtr, self.Xte, self.ytr, self.yte = cfg, Xtr, Xte, ytr, yte
        G = cfg.gates
        # θ=G/2はランダム初期化時の期待goodness(ゲート出力の平均≈0.5)に一致
        self.theta, self.tau = G / 2, float(np.sqrt(G))
        self.gneg = torch.Generator().manual_seed(cfg.seed * 31)
        self.X = ff_inputs(Xtr, ytr, cfg.n_class, cfg.ff_label_rep)  # 正例=配線基底
    def header(self):
        cfg = self.cfg
        return ("=== (A') Greedy layer-wise, Forward-Forward objective ==="
                + (f" [struct={cfg.ff_struct}]" if cfg.ff_struct > 0 else "")
                + (f" [window={cfg.window} commit={cfg.commit}]"
                   if cfg.window > 1 else "")
                + (f" [neg={cfg.ff_neg}"
                   + (f" warmup={cfg.ff_neg_warmup}" if cfg.ff_neg_warmup > 0 else "")
                   + (f" phases={cfg.ff_neg_phases}" if cfg.ff_neg_phases > 1 else "")
                   + (f" boost={cfg.ff_neg_boost}" if cfg.ff_neg_boost != 1.0 else "")
                   + "]" if cfg.ff_neg != "random" else ""))
    def begin(self, layers, d0):
        """負例ラベルを引き直し(正解を必ず避ける)、正例/負例プールを
        凍結プレフィックスで前進させる。窓の入力次元を返す"""
        cfg = self.cfg
        off = torch.randint(1, cfg.n_class, (len(self.ytr),),
                            generator=self.gneg).to(self.ytr.device)
        self.yneg = (self.ytr + off) % cfg.n_class    # 一様ランダムな誤りラベル
        self.Xn = ff_inputs(self.Xtr, self.yneg, cfg.n_class, cfg.ff_label_rep)
        _, self.pool_p = hard_pass(layers, self.X, cfg)
        _, self.pool_n = hard_pass(layers, self.Xn, cfg)
        return self.pool_p.shape[1]
    def make_layer(self, in_dim, d0, k):
        """最初の層(d0==0,k==0)だけ、--ff-struct>0 なら割合fのゲートを
        データ×ラベルに強制配線する。ラベルは入力の末尾に重畳されているので
        ラベル範囲は [Xtr幅, in_dim)。深い層はプールにラベルが無いので素のまま
        (層内のラベルアクセスは特徴経由になる = A案。各層再露出のB案は未実装)"""
        cfg = self.cfg
        seed = cfg.seed * 100 + d0 + k + 1
        if d0 == 0 and k == 0 and cfg.ff_struct > 0:
            data_hi = self.Xtr.shape[1]     # ラベル重畳前のデータ次元
            struct = (data_hi, data_hi, in_dim, cfg.ff_struct, cfg.seed * 300 + 1)
            return LogicLayer(in_dim, cfg.gates, seed, struct)
        return LogicLayer(in_dim, cfg.gates, seed)
    def _fit(self, win, pool_n, Xn, seed, epochs, opt, w=None):
        """FFロジスティック損失で窓を学習。wはサンプル別損失重み(None=一様)"""
        cfg, theta, tau = self.cfg, self.theta, self.tau
        def wmean(v, wb):
            return v.mean() if wb is None else (v * wb).sum() / wb.sum()
        def loss(idx):
            bp = self.pool_p if idx is None else self.pool_p[idx]
            bn = pool_n if idx is None else pool_n[idx]
            xp = self.X if idx is None else self.X[idx]
            xn = Xn if idx is None else Xn[idx]
            wb = w if (w is None or idx is None) else w[idx]
            terms = []
            for L in win:
                hp, hn = L(bp), L(bn)
                terms.append(wmean(F.softplus(-(hp.sum(1) - theta) / tau), wb)
                             + wmean(F.softplus((hn.sum(1) - theta) / tau), wb))
                bp, bn = next_pool(hp, xp, bp, cfg), next_pool(hn, xn, bn, cfg)
            return sum(terms) / len(terms)
        fit(loss, len(self.pool_p), cfg, seed, epochs, opt)
    def train(self, win, layers, d0):
        """warmup(ランダム負例で通常学習)→ 反復フェーズ(模試→負例再マイニング→
        重点復習)。マイニングは「模試の採点者」がいるときだけ発動: warmupありなら
        学習途中の窓自身が採点者になれる(層1でも可)、なしは凍結プレフィックスのみ"""
        cfg = self.cfg
        opt = torch.optim.Adam([p for L in win for p in L.parameters()], lr=cfg.lr)
        mining = cfg.ff_neg != "random" and (bool(layers) or cfg.ff_neg_warmup > 0)
        e1 = (cfg.epochs if not mining else
              max(1, int(cfg.epochs * cfg.ff_neg_warmup)) if cfg.ff_neg_warmup > 0
              else 0)
        seed_w = cfg.seed * 1000 + d0 + 1
        if e1:      # フェーズ0: 通常学習(ランダム負例)
            self._fit(win, self.pool_n, self.Xn, seed_w, e1, opt)
        rem = cfg.epochs - e1
        if mining and rem > 0:
            grader = win if cfg.ff_neg_warmup > 0 else []  # 学習中の窓 or 凍結前層
            K = cfg.ff_neg_phases
            for ph in range(K):
                ep = rem // K + (1 if ph < rem % K else 0)  # 残りをK等分
                if ep == 0:
                    continue
                g = self.mine(layers, grader)           # 模試(現在の窓で採点)
                pred = g.argmax(1)
                wrong = pred != self.ytr
                g.scatter_(1, self.ytr.view(-1, 1), float("-inf"))  # 正解を除外
                hard = g.argmax(1)      # 正解以外で最尤 = いま一番騙されるラベル
                if cfg.ff_neg == "hard":
                    yneg = hard
                elif cfg.ff_neg == "mix":   # 最難とランダムを半々
                    coin = (torch.rand(len(self.ytr), generator=self.gneg)
                            < 0.5).to(self.ytr.device)
                    yneg = torch.where(coin, hard, self.yneg)
                else:                       # review: 誤答は自分の誤答を負例に
                    yneg = torch.where(wrong, pred, self.yneg)  # 正解は通常のまま
                Xn = ff_inputs(self.Xtr, yneg, cfg.n_class, cfg.ff_label_rep)
                _, pool_n = hard_pass(layers, Xn, cfg)
                # boost: 誤答サンプルの損失をB倍に傾斜(苦手問題に時間を割く)
                w = (torch.where(wrong, float(cfg.ff_neg_boost), 1.0)
                     if cfg.ff_neg_boost != 1.0 else None)
                self._fit(win, pool_n, Xn, seed_w, ep, opt, w)
    @torch.no_grad()
    def mine(self, layers, win):
        """「模試」: 凍結プレフィックス(+学習途中の窓があればsoftで)を通した
        goodness行列 [N, n_class]。負例マイニングと誤答抽出に使う"""
        cfg = self.cfg
        good = []
        for c in range(cfg.n_class):
            Xc = ff_inputs(self.Xtr, torch.full_like(self.ytr, c),
                           cfg.n_class, cfg.ff_label_rep)
            gs = []
            for xc in Xc.split(8192):
                h, pool = hard_pass(layers, xc, cfg)
                for L in win:                    # 学習途中の窓はsoftで通す
                    h = L(pool)
                    pool = next_pool(h, xc, pool, cfg)
                gs.append(h.sum(1))
            good.append(torch.cat(gs))
        return torch.stack(good, 1)
    def commit(self, layers, L):
        """FFの離散化はプール前進のみ(beginで再計算するので状態は持たない)。
        全candidateラベル試行によるgoodnessプローブを返す"""
        a_tr = accuracy(self.counts(layers, self.Xtr, self.ytr), self.ytr)
        a_te = accuracy(self.counts(layers, self.Xte, self.yte), self.yte)
        return a_tr, a_te
    @torch.no_grad()
    def counts(self, layers, X, y):
        """クラス別goodness行列 [B, n_class](プローブ・推論・投票で共用)"""
        cfg = self.cfg
        good = []
        for c in range(cfg.n_class):
            Xc = ff_inputs(X, torch.full_like(y, c), cfg.n_class, cfg.ff_label_rep)
            h, _ = hard_pass(layers, Xc, cfg)
            good.append(h.sum(1))
        return torch.stack(good, 1)

def make_objective(Xtr, Xte, ytr, yte, cfg):
    cls = ForwardForward if cfg.objective == "ff" else GroupSum
    return cls(Xtr, Xte, ytr, yte, cfg)

# ----------------------------- (A) greedy layer-wise -----------------------------
def run_greedy(Xtr, Xte, ytr, yte, cfg):
    """greedy本体(目的関数に依存しない骨格): W層の窓をsoftで学習し、先頭J層を
    離散化・凍結して窓をスライド。プローブが頭打ちになったら深さを確定する"""
    obj = make_objective(Xtr, Xte, ytr, yte, cfg)
    W, J = cfg.window, cfg.commit
    print(obj.header()
          + (" [carry]" if cfg.carry else "")
          + (" [skip-all wiring]" if cfg.skip_all else
             " [skip-input wiring]" if cfg.skip_input else ""))
    layers, best_acc, best_depth, since_best = [], -1.0, 0, 0
    t0 = time.time()
    stop = False
    carry = []   # --carry: コミットされなかった先読み層を次スライドへ受け継ぐ足場方式
    while not stop and len(layers) < cfg.max_layers:
        d0 = len(layers)
        # 凍結済みプレフィックスの上にW層の窓を新規作成(スライドごとに再計画。
        # コミットされなかったlookahead層は捨てる = receding horizon)
        win, in_dim = [], obj.begin(layers, d0)
        for k in range(W):
            if k < len(carry):
                win.append(carry[k])   # 受け継いだ層(重み保持、no-skipでin_dim=G一致)
            else:
                win.append(obj.make_layer(in_dim, d0, k).to(cfg.device))
            in_dim = (in_dim + cfg.gates if cfg.skip_all else
                      obj.X.shape[1] + cfg.gates if cfg.skip_input else cfg.gates)
        obj.train(win, layers, d0)
        carry = win[J:] if cfg.carry else []
        for L in win[:J]:  # 窓の先頭J層だけ離散化して凍結(HARDビット上で確定)
            layers.append(L)
            a_tr, a_te = obj.commit(layers, L)
            print(f"  layer {len(layers)}: {obj.kind} probe"
                  f"  train={a_tr:.4f}  test={a_te:.4f}")
            if a_te > best_acc + 1e-4:
                best_acc, best_depth, since_best = a_te, len(layers), 0
            else:
                since_best += 1
                if since_best >= cfg.patience:
                    print(f"  -> stop: no improvement for {cfg.patience} layers")
                    stop = True
                    break
            if len(layers) >= cfg.max_layers:
                break
    print(f"  {obj.tag}: best {obj.kind} test acc = {best_acc:.4f}"
          f" at depth {best_depth}  ({time.time() - t0:.0f}s)\n")
    return layers[:best_depth], best_acc, best_depth

# ----------------------------- (D) ensemble of greedy networks -----------------------------
def run_ensemble(Xtr, Xte, ytr, yte, cfg):
    """独立に学習したM本のネットワーク(シードのみ変える)を横に並べて投票。
    soft vote: 各メンバーのクラス別カウント(GroupSumカウント / FF goodness)を
               合算してargmax(groupsumでは幅M倍readoutと数学的に等価)
    majority : 各メンバーのargmaxで多数決。同数タイは合算カウントの大きい方
               (0.99倍で正規化したスコアを足すので票数の優劣は覆らない)"""
    M, base_seed = cfg.ensemble, cfg.seed
    members, member_acc, depths = [], [], []
    for m in range(M):
        cfg.seed = base_seed + m
        print(f"--- ensemble member {m + 1}/{M} (seed {cfg.seed}) ---")
        layers, acc, depth = run_greedy(Xtr, Xte, ytr, yte, cfg)
        members.append(layers); member_acc.append(acc); depths.append(depth)
    cfg.seed = base_seed
    obj = make_objective(Xtr, Xte, ytr, yte, cfg)
    counts = torch.stack([obj.counts(ls, Xte, yte)
                          for ls in members])                    # [M, B, n_class]
    soft_acc = accuracy(counts.sum(0), yte)
    votes = F.one_hot(counts.argmax(-1), cfg.n_class).sum(0).float()
    c = counts.sum(0)
    t = (c - c.amin(1, keepdim=True)) / (c.amax(1, keepdim=True)
                                         - c.amin(1, keepdim=True) + 1e-9)
    maj_acc = accuracy(votes + 0.99 * t, yte)
    label = "goodness" if cfg.objective == "ff" else "GroupSum"
    print(f"=== (D) Ensemble vote over {M} members ===")
    print(f"  member acc: {' / '.join(f'{a:.4f}' for a in member_acc)}"
          f"  (mean {float(np.mean(member_acc)):.4f}, depths {depths})")
    print(f"  soft vote (summed {label} counts) = {soft_acc:.4f}")
    print(f"  majority vote (count tie-break)    = {maj_acc:.4f}\n")
    return members, member_acc, depths, soft_acc, maj_acc

# ----------------------------- (B) end-to-end baseline -----------------------------
def run_e2e(Xtr, Xte, ytr, yte, depth, cfg):
    print(f"=== (B) End-to-end backprop baseline, {depth} layers ===")
    tau = float(np.sqrt(cfg.gates / cfg.n_class))
    net = nn.ModuleList([LogicLayer(Xtr.shape[1] if i == 0 else cfg.gates,
                                    cfg.gates, seed=cfg.seed * 200 + i)
                         for i in range(depth)]).to(cfg.device)
    opt = torch.optim.Adam([p for L in net for p in L.parameters()], lr=cfg.lr)
    epochs = min(cfg.epochs * depth, cfg.e2e_max_epochs)
    t0 = time.time()
    g = torch.Generator().manual_seed(cfg.seed * 777)
    for _ in range(epochs):
        if cfg.batch and cfg.batch < len(Xtr):
            for idx in torch.randperm(len(Xtr), generator=g).split(cfg.batch):
                idx = idx.to(Xtr.device)
                opt.zero_grad()
                h = Xtr[idx]
                for L in net:
                    h = L(h)
                F.cross_entropy(group_sum(h, cfg.n_class, tau), ytr[idx]).backward()
                opt.step()
        else:
            opt.zero_grad()
            h = Xtr
            for L in net:
                h = L(h)
            F.cross_entropy(group_sum(h, cfg.n_class, tau), ytr).backward()
            opt.step()
    with torch.no_grad():
        chunks = []
        for c in Xte.split(4096):   # chunked soft eval bounds the [B, G, 16] temporary
            for L in net:
                c = L(c)
            chunks.append(c)
        hs = torch.cat(chunks)
        hh = Xte
        for L in net:
            hh = hard_batched(L, hh)
    soft = accuracy(group_sum(hs, cfg.n_class, tau), yte)
    hard = accuracy(group_sum(hh, cfg.n_class, tau), yte)
    print(f"  soft={soft:.4f}  discretized={hard:.4f}  gap={soft - hard:+.4f}"
          f"  ({epochs} epochs, {time.time() - t0:.0f}s)")
    print(f"  float logits during training: greedy={cfg.gates * 16:,} (1 layer)"
          f"  vs  e2e={cfg.gates * 16 * depth:,} (x{depth})\n")
    return soft, hard

# ----------------------------- (C) logic simplification -----------------------------
def simplify(layers, Xte, yte, cfg):
    """Constant folding, pass-through/NOT reduction, duplicate merge,
    dead-gate elimination. Verifies the simplified circuit is bit-identical."""
    print("=== (C) Logic simplification of the greedy hard network ===")
    in_bits, G, D = Xte.shape[1], cfg.gates, len(layers)
    tau = float(np.sqrt(G / cfg.n_class))
    net, base = [], in_bits
    skip = getattr(cfg, "skip_input", False)
    dense = getattr(cfg, "skip_all", False)
    for li, L in enumerate(layers):
        prev = base - G  # global id of the previous layer's first gate (li > 0)
        def src(j, li=li, prev=prev):
            if li == 0:
                return j                                  # wired to input bits
            if dense:                                     # pool = [input || all layers]:
                return j                                  #   pool order == global id order
            if skip:                                      # pool = [input || prev layer]
                return j if j < in_bits else prev + (j - in_bits)
            return prev + j                               # pool = prev layer only
        fn = L.logits.argmax(-1).tolist()
        net.append([(base + i, src(int(L.ia[i])), src(int(L.ib[i])), fn[i])
                    for i in range(G)])
        base += G
    total_before = D * G
    final_ids = [g[0] for g in net[-1]]

    # reference evaluation
    vals = {i: Xte[:, i] for i in range(in_bits)}
    for layer in net:
        for gid, ia, ib, fn in layer:
            vals[gid] = f16(fn, vals[ia], vals[ib])
    ref = torch.stack([vals[i] for i in final_ids], 1)
    acc0 = accuracy(group_sum(ref, cfg.n_class, tau), yte)

    resolve = {}
    def res(n):
        while True:
            r = resolve.get(n)
            if r is None:
                return ('node', n)
            if r[0] == 'const':
                return r
            n = r[1]

    c = dict(const=0, passthru=0, dup=0)
    kept = []
    for layer in net:
        seen, out = {}, []
        for gid, ia0, ib0, fn in layer:
            ra, rb = res(ia0), res(ib0)
            if ra[0] == 'node' and rb[0] == 'node' and ra[1] > rb[1]:
                ra, rb, fn = rb, ra, SWAP[fn]
            if ra[0] == 'const' and rb[0] == 'const':
                v = int(f16(fn, torch.tensor(float(ra[1])),
                            torch.tensor(float(rb[1]))).item())
                resolve[gid] = ('const', v); c['const'] += 1; continue
            if ra[0] == 'const' or rb[0] == 'const':
                cv = float((ra if ra[0] == 'const' else rb)[1])
                other = (rb if ra[0] == 'const' else ra)[1]
                z, o = torch.tensor(0.), torch.tensor(1.)
                if ra[0] == 'const':
                    v0, v1 = int(f16(fn, torch.tensor(cv), z).item()), \
                             int(f16(fn, torch.tensor(cv), o).item())
                else:
                    v0, v1 = int(f16(fn, z, torch.tensor(cv)).item()), \
                             int(f16(fn, o, torch.tensor(cv)).item())
                if v0 == v1:
                    resolve[gid] = ('const', v0); c['const'] += 1; continue
                if (v0, v1) == (0, 1):
                    resolve[gid] = ('node', other); c['passthru'] += 1; continue
                ra = rb = ('node', other); fn = 12       # NOT(other)
            ia, ib = ra[1], rb[1]
            if fn == 3:
                resolve[gid] = ('node', ia); c['passthru'] += 1; continue
            if fn == 5:
                resolve[gid] = ('node', ib); c['passthru'] += 1; continue
            if fn == 0:
                resolve[gid] = ('const', 0); c['const'] += 1; continue
            if fn == 15:
                resolve[gid] = ('const', 1); c['const'] += 1; continue
            key = (ia, ib, fn)
            if key in seen:
                resolve[gid] = ('node', seen[key]); c['dup'] += 1; continue
            seen[key] = gid
            out.append((gid, ia, ib, fn))
        kept.append(out)

    live = set()
    for gid in final_ids:
        r = res(gid)
        if r[0] == 'node' and r[1] >= in_bits:
            live.add(r[1])
    for layer in reversed(kept):
        for gid, ia, ib, fn in layer:
            if gid in live:
                if ia >= in_bits: live.add(ia)
                if ib >= in_bits: live.add(ib)
    kept = [[g for g in l if g[0] in live] for l in kept]
    total_after = sum(len(l) for l in kept)
    dead = total_before - c['const'] - c['passthru'] - c['dup'] - total_after

    # verify simplified circuit
    vals2 = {i: Xte[:, i] for i in range(in_bits)}
    for layer in kept:
        for gid, ia, ib, fn in layer:
            vals2[gid] = f16(fn, vals2[ia], vals2[ib])
    cols = []
    for gid in final_ids:
        r = res(gid)
        cols.append(torch.full((len(Xte),), float(r[1]))
                    if r[0] == 'const' else vals2[r[1]])
    out = torch.stack(cols, 1)
    acc1 = accuracy(group_sum(out, cfg.n_class, tau), yte)
    identical = torch.equal(ref, out)

    print(f"  gates: {total_before:,} -> {total_after:,}"
          f" ({100 * total_after / total_before:.1f}%)")
    print(f"  const-folded={c['const']}, pass-through={c['passthru']},"
          f" duplicates={c['dup']}, dead={dead}")
    print(f"  test acc {acc0:.4f} -> {acc1:.4f}, outputs identical = {identical}\n")
    assert identical, "simplification changed the function!"
    return total_before, total_after

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
    p.add_argument("--dataset", choices=["digits", "mnist"], default="digits",
                   help="digits: sklearn 8x8 (CPU-friendly). mnist: 28x28, 70k"
                        " samples (GPU + --batch recommended)")
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
    cfg.n_class = 10
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in load_data(cfg.dataset)]
    print(f"data: {cfg.dataset}, {Xtr.shape[0]} train / {Xte.shape[0]} test,"
          f" {Xtr.shape[1]} input bits  (device={cfg.device}"
          + (f", batch={cfg.batch}" if cfg.batch else "") + ")\n")

    # (FFの簡略化検証: 入力はラベル重畳込み。簡略化器が表示するaccはGroupSum
    #  読み出しなのでFFの精度ではないが、ビット等価性の検証はそのまま有効)
    Xte_s = (ff_inputs(Xte, yte, cfg.n_class, cfg.ff_label_rep)
             if cfg.objective == "ff" else Xte)

    if cfg.ensemble > 1:
        members, member_acc, depths, soft_acc, maj_acc = run_ensemble(
            Xtr, Xte, ytr, yte, cfg)
        e2e_soft = e2e_hard = None
        if not cfg.skip_e2e:
            e2e_soft, e2e_hard = run_e2e(Xtr, Xte, ytr, yte,
                                         cfg.e2e_depth or depths[0], cfg)
        before = after = 0
        for ls in members:  # メンバーごとに簡略化+ビット等価検証
            b, a = simplify([L.cpu() for L in ls], Xte_s.cpu(), yte.cpu(), cfg)
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
    # residual: readout sums EVERY layer's class bits, so dead-gate elimination
    # (which prunes gates not feeding the LAST layer) would drop contributing
    # gates. Skip until simplify treats all layers as outputs (future work).
    if cfg.group_residual:
        print("=== (C) simplification skipped (residual: all-layer readout) ===\n")
        before = after = 0
    else:
        before, after = simplify([L.cpu() for L in layers], Xte_s.cpu(), yte.cpu(), cfg)

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
