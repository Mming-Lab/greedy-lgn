"""畳み込み論理層(Phase 2): 重み共有カーネル+ORプーリング。
構成は畳み込みDLGN(Petersen et al. 2024)に従う — C個の「論理カーネル」
(深さTの完全二分木、葉はk×k×Cin窓からランダム抽選)を全位置に重み共有で
適用し、2×2プーリング(binaryではmax=OR)で受容野を成長させる。
うちの問いは「greedy層ごと学習(+残差readout)と畳み込みの組合せ」。
Phase 1(--local、局所性のみ)がMNISTで否定された穴 — 受容野が育たない —
をプーリングが埋め、重み共有がパラメータあたりの実効サンプルを増やす。
学習ロジットはC×(2^T-1)ゲート分だけ、展開回路はH×W×C×(2^T-1)ゲート
(--recurと同じく学習予算と推論面積が分離する)。simplifyは未対応(将来課題)。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from core import group_sum, accuracy, fit, make_stop_check

# 16ゲートは全て {1, a, b, ab} の線形結合: gate_k = M[k]·[1,a,b,ab]。
# softmax重みをこの基底に畳む(16→4)ことで [.,16] スタックを作らずに
# Σ_k w_k gate_k(a,b) = c0 + c1 a + c2 b + c3 ab と計算できる(all16展開と
# 数学的に厳密同値、ピークメモリ約1/16でOOM回避)。順序はcore.all16に一致。
GATE_BASIS = torch.tensor([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0],
    [0, 0, 1, -1], [0, 0, 1, 0], [0, 1, 1, -2], [0, 1, 1, -1],
    [1, -1, -1, 1], [1, -1, -1, 2], [1, 0, -1, 0], [1, 0, -1, 1],
    [1, -1, 0, 0], [1, -1, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0],
], dtype=torch.float32)

# soft学習でチャンクあたりに許す葉テンソル [B, Cc, L, HW] の要素数上限
# (チャンネルチャンクの予算。hard_chunkedのバッチ予算と同水準)
SOFT_BUDGET = 120_000_000


class ConvLogicLayer(nn.Module):
    """1つの畳み込み論理層。入力 [B, Cin*H*W](チャンネルメジャー平坦)→
    出力 [B, C*Hp*Wp](プーリング後)。soft(forward)/hard 両対応"""
    def __init__(self, Cin, H, W, C, k, tree, pool, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Cin, self.H, self.W, self.C, self.k, self.pool = Cin, H, W, C, k, pool
        self.L = 2 ** tree                                  # 葉の数
        # 各チャンネルの葉配線: k*k*Cin 窓内のランダムなオフセット
        self.register_buffer("leaf",
                             torch.randint(0, Cin * k * k, (C, self.L), generator=g))
        self.logits = nn.Parameter(torch.randn(C, self.L - 1, 16, generator=g))
        self.register_buffer("basis", GATE_BASIS.clone())   # [16, 4]
        self.Hp, self.Wp = (H // pool, W // pool) if pool > 1 else (H, W)
    def _leaves(self, x, leaf):
        """[B, Cin*H*W] → 葉の値 [B, Cc, L, H*W](zero-padding=境界外は0ビット)。
        leaf=[Cc, L] はチャンネルチャンク(全体は self.leaf)"""
        B = x.shape[0]
        u = F.unfold(x.view(B, self.Cin, self.H, self.W),
                     self.k, padding=self.k // 2)           # [B, Cin*k*k, HW]
        return u[:, leaf.reshape(-1), :].view(B, leaf.shape[0], self.L, -1)
    def _poolflat(self, v):
        """根の値 [B, Cc, HW] → プーリングして平坦 [B, Cc*Hp*Wp]"""
        B = v.shape[0]
        if self.pool > 1:
            v = F.max_pool2d(v.view(B, -1, self.H, self.W), self.pool)
        return v.reshape(B, -1)
    def _combine(self, a, b, coeff):
        """a,b: [B,C,n,HW]、coeff: [C,n,4] → c0 + c1 a + c2 b + c3 ab。
        16ゲートの重み和を [.,16] スタックなしで計算(メモリの要)"""
        c = [coeff[..., i].unsqueeze(0).unsqueeze(-1) for i in range(4)]  # [1,C,n,1]
        return c[0] + c[1] * a + c[2] * b + c[3] * (a * b)
    def _reduce(self, v, coeff_all):
        """木を1段ずつ縮約。coeff_all: [C, L-1, 4]"""
        i0 = 0
        while v.shape[2] > 1:
            n = v.shape[2] // 2
            v = self._combine(v[:, :, 0::2], v[:, :, 1::2], coeff_all[:, i0:i0 + n])
            i0 += n
        return self._poolflat(v.squeeze(2))
    def _soft(self, x, coeff, leaf):
        return self._reduce(self._leaves(x, leaf), coeff)
    def forward(self, x):                                   # soft(学習)
        coeff = F.softmax(self.logits, -1) @ self.basis     # [C, L-1, 4](16->4)
        if torch.is_grad_enabled():
            # 勾配チェックポイント: 葉テンソル[B,Cc,L,HW]と各段のa,b,abを順伝播で
            # 保持せずbackwardで再計算(6GBでMNIST 28x28を通すための要。入力xと
            # 係数coeffだけ保持)。use_reentrant=Falseでno-grad入力に対応。
            # さらにチャンネルをチャンク分割(バッチ分割のhard_chunkedと同じ予算制)
            # してチャンクごとにcheckpoint: backward再計算時のピークを定数化する。
            # チャンネルは集計まで完全独立(葉配線・係数・出力ともCで分割可能、
            # 勾配は各チャンクのlogitsスライスに閉じ、入力xは凍結ビットで勾配
            # なし)。等価性の正直な範囲(2026-07-14実測): forward出力は分割数に
            # よらずビット一致。ただしlogits勾配はB×HW縮約の順序がテンソル形状で
            # 変わり最終ビットの丸めが変わりうる(相対~1e-6、数学的には同値)。
            # 既存のピン留め構成(digits)は全てCc>=Cの1チャンク=完全にビット等価で、
            # マルチチャンクは従来OOMで走れなかった構成でのみ発動する。
            # 同じ構成・同じ予算ならチャンク数は決定的なので再現性は保たれる
            Cc = max(1, SOFT_BUDGET // max(1, x.shape[0] * self.L * self.H * self.W))
            if Cc >= self.C:
                return checkpoint(self._soft, x, coeff, self.leaf,
                                  use_reentrant=False)
            outs = [checkpoint(self._soft, x, coeff[c0:c0 + Cc],
                               self.leaf[c0:c0 + Cc], use_reentrant=False)
                    for c0 in range(0, self.C, Cc)]
            return torch.cat(outs, 1)   # チャンネルメジャー平坦なのでcat=C連結
        return self._soft(x, coeff, self.leaf)
    @torch.no_grad()
    def hard(self, x):                                      # 離散化(推論)
        coeff = self.basis[self.logits.argmax(-1)]          # [C, L-1, 4](選択ゲート)
        return self._reduce(self._leaves(x, self.leaf), coeff)


def hard_chunked(L, x, budget=120_000_000):
    """convのhardをバッチ分割で評価。葉テンソル [chunk, C, L, HW] がMNISTでは
    巨大(chunk2048で3.2GB)なので、chunk*C*L*HW を budget 以下に抑える。
    x はuint8プールでも可(チャンク単位でfloatへ。float入力では恒等・無コピー)"""
    per = max(1, L.C * L.L * L.H * L.W)
    chunk = max(64, budget // per)
    return torch.cat([L.hard(c.float()) for c in x.split(chunk)])


class ConvGroupSum:
    """畳み込み版のGroupSum目的。プールは特徴マップの平坦ビット列で、
    形状 (Cin, H, W) を層のコミットごとに前進させる。残差readout対応。
    読み出しは全出力ビットのGroupSum(τ=√(bits/n_class)を層ごとの幅に追従)"""
    kind, tag = "hard", "greedy-conv"
    def __init__(self, Xtr, Xte, ytr, yte, cfg):
        self.cfg, self.ytr, self.yte = cfg, ytr, yte
        self.X, self.Xte = Xtr, Xte      # run_greedyのskip分岐用(convはno-skip限定)
        w, npix = (8, 64) if cfg.dataset == "digits" else (28, 784)
        self.shape = (Xtr.shape[1] // npix, w, w)           # (Cin=面数, H, W)
        # プールはハード0/1ビットなのでuint8常駐(groupsum.pyと同じOOM対策。
        # 学習・hard評価の直前にバッチ/チャンク単位でfloatへ=ビット等価)
        self.pool_tr = Xtr.to(torch.uint8)
        self.pool_te = Xte.to(torch.uint8)
        if cfg.group_residual:
            self.accum_tr = torch.zeros(len(ytr), cfg.n_class, device=cfg.device)
            self.accum_te = torch.zeros(len(yte), cfg.n_class, device=cfg.device)
    def header(self):
        cfg = self.cfg
        Cin, H, W = self.shape
        sched = getattr(cfg, "conv_sched", None)
        cdesc = f"C={','.join(map(str, sched))}" if sched else f"C={cfg.conv}"
        return (f"=== (A''') Greedy layer-wise, convolutional logic kernels"
                f" ({cdesc}, k={cfg.conv_k}, tree={cfg.conv_tree},"
                f" pool={cfg.conv_pool}) ==="
                + (" [residual]" if cfg.group_residual else ""))
    def begin(self, layers, d0):
        return self.pool_tr.shape[1]
    def _channels(self, d0):
        """層d0のチャンネル数。--conv-sched でスケジュール指定(逆ファンネル等)、
        未指定なら全層 cfg.conv 一定。範囲外の深さは最後の値を使い回す
        (生物のV1: LGN→V1で17〜40倍展開=逆ファンネルが着想元。--conv-sched 128,64,32)"""
        sched = getattr(self.cfg, "conv_sched", None)
        if sched:
            return sched[min(d0, len(sched) - 1)]
        return self.cfg.conv
    def make_layer(self, in_dim, d0, k):
        cfg = self.cfg
        Cin, H, W = self.shape
        pool = cfg.conv_pool if min(H, W) >= 2 * cfg.conv_pool else 1
        C = self._channels(d0)
        L = ConvLogicLayer(Cin, H, W, C, cfg.conv_k, cfg.conv_tree, pool,
                           seed=cfg.seed * 100 + d0 + k + 1)
        self._pending_shape = (C, L.Hp, L.Wp)
        return L
    def _tau(self, bits):
        return float(np.sqrt(bits / self.cfg.n_class))
    def train(self, win, layers, d0):
        """convはwindow=1限定(win=[L])。損失は当層出力(プーリング後)のGroupSum CE、
        残差時は凍結累積をオフセットに"""
        cfg, L = self.cfg, win[0]
        opt = torch.optim.Adam(L.parameters(), lr=cfg.lr)
        def loss(idx):
            # uint8常駐プールをこのバッチ分だけfloatへ(0/1なので厳密・ビット等価)
            x = (self.pool_tr if idx is None else self.pool_tr[idx]).float()
            y = self.ytr if idx is None else self.ytr[idx]
            h = L(x)
            s = group_sum(h, cfg.n_class, self._tau(h.shape[1]))
            if cfg.group_residual:
                s = (self.accum_tr if idx is None else self.accum_tr[idx]) + s
            return F.cross_entropy(s, y)
        fit(loss, len(self.ytr), cfg, cfg.seed * 1000 + d0 + 1, cfg.epochs, opt,
            stop_check=make_stop_check(win, cfg, d0, self))
    def commit(self, layers, L):
        cfg = self.cfg
        h_tr = hard_chunked(L, self.pool_tr)
        h_te = hard_chunked(L, self.pool_te)
        tau = self._tau(h_tr.shape[1])
        s_tr = group_sum(h_tr, cfg.n_class, tau)
        s_te = group_sum(h_te, cfg.n_class, tau)
        if cfg.group_residual:
            self.accum_tr = self.accum_tr + s_tr
            self.accum_te = self.accum_te + s_te
            s_tr, s_te = self.accum_tr, self.accum_te
        a_tr, a_te = accuracy(s_tr, self.ytr), accuracy(s_te, self.yte)
        # hardビット(厳密に0.0/1.0)で前進、常駐はuint8
        self.pool_tr = h_tr.to(torch.uint8)
        self.pool_te = h_te.to(torch.uint8)
        self.shape = self._pending_shape
        return a_tr, a_te
    @torch.no_grad()
    def counts(self, layers, X, y):
        """アンサンブル投票用(残差なら全層累積、そうでなければ最終層のカウント)"""
        cfg, x = self.cfg, X
        acc = torch.zeros(len(X), cfg.n_class, device=X.device)
        for L in layers:
            x = hard_chunked(L, x)
            if cfg.group_residual:
                acc = acc + group_sum(x, cfg.n_class, self._tau(x.shape[1]))
        return acc if cfg.group_residual else group_sum(x, cfg.n_class, 1.0)
