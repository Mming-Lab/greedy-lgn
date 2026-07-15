"""row-sequential再帰(RDDLGN流の時系列状態をgreedyに移植した実験)。
画像を行ごとに提示し、各greedy層=再帰セル: s_t = L([x_t ; s_{t-1}])。
学習はTステップのBPTT(soft)、コミット後はhardビットの状態系列を次層の入力に
する(凍結済み層の状態はすべて実ビット)。配線(連結)と読み出し(GroupSum)は
RDDLGN (arXiv:2508.06097) と同型 — 新規性は主張しない。うちの問いは
「greedy層ごと学習 × 時系列再帰」がどう振る舞うか。
正直な注意: 学習中のBPTTはsoft状態を流すので、再帰層内の時間方向には
離散化ギャップが再出現しうる(凍結層間のギャップ0はこれまで通り)。"""
import numpy as np
import torch
import torch.nn.functional as F
from core import (GATE_A, LogicLayer, group_sum, accuracy, hard_batched, fit,
                  make_stop_check, img_shape)


class SeqGroupSum:
    """時系列版GroupSum目的。プールの代わりに「ステップ別入力系列」を保持し、
    層のコミットごとに系列を(その層のhard状態系列に)置き換えて前進する"""
    kind, tag = "hard", "greedy-seq"
    def __init__(self, Xtr, Xte, ytr, yte, cfg):
        self.cfg, self.ytr, self.yte = cfg, ytr, yte
        self.X, self.Xte = Xtr, Xte     # run_greedyのskip分岐用(seqはno-skip限定)
        self.tau = float(np.sqrt(cfg.gates / cfg.n_class))
        # thermometer連結 [(X>t1),(X>t2),...] から行tのビット列を取り出す添字:
        # 各閾値ブロック内で t*w..t*w+w の画素 → 1行 = 面数*w ビット
        # (面数は入力幅から動的に導出 — --thresholds で面数が変わっても追従)
        w, npix = img_shape(cfg.dataset)
        planes = Xtr.shape[1] // npix
        self.T = npix // w
        self.row_idx = [torch.tensor([k * npix + t * w + i
                                      for k in range(planes) for i in range(w)],
                                     device=Xtr.device)
                        for t in range(self.T)]
        self.seq_tr = [Xtr[:, ix] for ix in self.row_idx]
        self.seq_te = [Xte[:, ix] for ix in self.row_idx]
    def header(self):
        cfg = self.cfg
        return ("=== (A'') Greedy layer-wise, row-sequential recurrent"
                f" (T={self.T}, {self.seq_tr[0].shape[1]} bits/step) ==="
                + (f" [warm-start={cfg.warm_start}]" if cfg.warm_start > 0 else ""))
    def begin(self, layers, d0):    # 窓の入力次元 = ステップ入力幅 + 状態幅
        return self.seq_tr[0].shape[1] + self.cfg.gates
    def make_layer(self, in_dim, d0, k):
        # seqのwarm-startは「前層状態(連結の先頭スライス)」への恒等にする。
        # LogicLayer標準のwarm(末尾恒等)だと末尾=自分の状態なので「ゼロ初期値の
        # 自状態を保持し続ける」自己参照ループになり崩壊する(digitsで82%→27%を
        # 実測)。先頭恒等=「下の層の状態のコピーから始める」で静的warm-startと
        # 同じ意味論。層1は先頭が行入力(幅<gates)なので素のまま
        cfg = self.cfg
        L = LogicLayer(in_dim, cfg.gates, seed=cfg.seed * 100 + d0 + k + 1)
        if cfg.warm_start > 0 and (d0 + k) > 0:
            with torch.no_grad():
                L.ia.copy_(torch.arange(cfg.gates))       # gate i <- 前層状態ビットi
                L.logits[:, GATE_A] += cfg.warm_start
        return L
    def train(self, win, layers, d0):
        """再帰セルをTステップのBPTTで学習(loss=最終状態のGroupSum CE)。
        seqはwindow=1限定なのでwin=[L]"""
        cfg, L = self.cfg, win[0]
        opt = torch.optim.Adam(L.parameters(), lr=cfg.lr)
        def loss(idx):
            xs = (self.seq_tr if idx is None
                  else [x[idx] for x in self.seq_tr])
            y = self.ytr if idx is None else self.ytr[idx]
            s = torch.zeros(len(y), cfg.gates, device=cfg.device)
            for x in xs:
                s = L(torch.cat([x, s], 1))
            return F.cross_entropy(group_sum(s, cfg.n_class, self.tau), y)
        fit(loss, len(self.ytr), cfg, cfg.seed * 1000 + d0 + 1, cfg.epochs, opt,
            stop_check=make_stop_check(win, cfg, d0, self))
    @torch.no_grad()
    def _roll(self, L, xs, n):
        """離散化済みセルのhardロールアウト。状態系列(全ステップ)を返す"""
        s, out = torch.zeros(n, self.cfg.gates, device=self.cfg.device), []
        for x in xs:
            s = hard_batched(L, torch.cat([x, s], 1))
            out.append(s)
        return out
    def commit(self, layers, L):
        """セルを離散化・凍結し、系列をhard状態系列に置き換えて前進。
        プローブは最終状態のGroupSum"""
        cfg = self.cfg
        self.seq_tr = self._roll(L, self.seq_tr, len(self.ytr))
        self.seq_te = self._roll(L, self.seq_te, len(self.yte))
        a_tr = accuracy(group_sum(self.seq_tr[-1], cfg.n_class, self.tau), self.ytr)
        a_te = accuracy(group_sum(self.seq_te[-1], cfg.n_class, self.tau), self.yte)
        return a_tr, a_te
    @torch.no_grad()
    def counts(self, layers, X, y):
        """アンサンブル投票用: 生入力から全層をhardで通した最終状態の整数カウント"""
        xs = [X[:, ix] for ix in self.row_idx]
        for L in layers:
            xs = self._roll(L, xs, len(X))
        return group_sum(xs[-1], self.cfg.n_class, 1.0)
