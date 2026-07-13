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
from core import all16, group_sum, accuracy, fit, make_stop_check


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
        self.Hp, self.Wp = (H // pool, W // pool) if pool > 1 else (H, W)
    def _leaves(self, x):
        """[B, Cin*H*W] → 葉の値 [B, C, L, H*W](zero-padding=境界外は0ビット)"""
        B = x.shape[0]
        u = F.unfold(x.view(B, self.Cin, self.H, self.W),
                     self.k, padding=self.k // 2)           # [B, Cin*k*k, HW]
        return u[:, self.leaf.view(-1), :].view(B, self.C, self.L, -1)
    def _poolflat(self, v):
        """根の値 [B, C, HW] → プーリングして平坦 [B, C*Hp*Wp]"""
        B = v.shape[0]
        if self.pool > 1:
            v = F.max_pool2d(v.view(B, self.C, self.H, self.W), self.pool)
        return v.reshape(B, -1)
    def forward(self, x):                                   # soft(学習)
        v = self._leaves(x)
        w = F.softmax(self.logits, -1)                      # [C, L-1, 16]
        i0 = 0
        while v.shape[2] > 1:                               # 木を1段ずつ縮約
            n = v.shape[2] // 2
            g16 = all16(v[:, :, 0::2], v[:, :, 1::2])       # [B, C, n, HW, 16]
            v = (g16 * w[:, i0:i0 + n].unsqueeze(0).unsqueeze(3)).sum(-1)
            i0 += n
        return self._poolflat(v.squeeze(2))
    @torch.no_grad()
    def hard(self, x):                                      # 離散化(推論)
        v = self._leaves(x)
        sel = self.logits.argmax(-1)                        # [C, L-1]
        i0 = 0
        while v.shape[2] > 1:
            n, B, HW = v.shape[2] // 2, v.shape[0], v.shape[3]
            g16 = all16(v[:, :, 0::2], v[:, :, 1::2])
            s = sel[:, i0:i0 + n].view(1, self.C, n, 1, 1).expand(B, -1, -1, HW, 1)
            v = g16.gather(-1, s).squeeze(-1)
            i0 += n
        return self._poolflat(v.squeeze(2))


def hard_chunked(L, x, chunk=2048):
    """convのhardをバッチ分割で(all16の一時テンソル [B,C,n,HW,16] を抑える)"""
    return torch.cat([L.hard(c) for c in x.split(chunk)])


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
        self.pool_tr, self.pool_te = Xtr, Xte
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
            x = self.pool_tr if idx is None else self.pool_tr[idx]
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
        self.pool_tr, self.pool_te = h_tr, h_te             # hardビットで前進
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
