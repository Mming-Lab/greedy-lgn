"""GroupSum目的(主線): 各層のGroupSum+CE readoutでローカル学習。
residual(--group-residual)・boost(--group-boost)・warm-start(--warm-start)は
このクラスのフラグ分岐として実装されたモード(クラス抽出はビット等価を崩す
リスクの割に利得がないので意図的にしていない)。"""
import numpy as np
import torch
import torch.nn.functional as F
from core import (LogicLayer, group_sum, group_mean, accuracy, next_pool,
                  hard_batched, hard_pass, fit, make_stop_check, reps)

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
        # プールはハードな0/1ビットなのでuint8で常駐(float32比1/4。MNIST 8000
        # ゲートの常駐プール2.5GB→0.6GBでissue #9のOOMを解消)。0/1のuint8⇄float
        # 往復は厳密に可逆で、学習・hard評価はミニバッチ/チャンク単位でfloatへ
        # 戻すので下流は同一のfloat値を受け取る=ビット等価。
        # X_u8/Xte_u8 はskip時のプール構築用(self.X/Xteはwindow>1のloss内で
        # softなfloat出力とcatするためfloatのまま残す)
        self.X_u8, self.Xte_u8 = Xtr.to(torch.uint8), Xte.to(torch.uint8)
        self.pool_tr, self.pool_te = self.X_u8, self.Xte_u8
        self.tau = float(np.sqrt(cfg.gates / cfg.n_class))
        # --group-residual: 凍結層のクラススコアを累積し、各層は「前層までの累積
        # 予測」を固定オフセットに残差を埋めるよう学習(ブースティング)。累積は
        # 全層のクラス別ビット総和 = 純論理回路のまま
        if cfg.group_residual:
            self.accum_tr = torch.zeros(len(ytr), cfg.n_class, device=cfg.device)
            self.accum_te = torch.zeros(len(yte), cfg.n_class, device=cfg.device)
        # --local: 位置の帳簿。入力ビットの画素位置は plane*npix+pixel → idx%npix。
        # プール前進とともに更新する(層出力の位置=そのゲートの割当位置)
        if getattr(cfg, "local", 0) > 0:
            self._w, self._npix = (8, 64) if cfg.dataset == "digits" else (28, 784)
            self.pos_input = [i % self._npix for i in range(Xtr.shape[1])]
            self.pos_pool = list(self.pos_input)
            self._pending_pos = None
    def header(self):
        cfg = self.cfg
        return ("=== (A) Greedy layer-wise: local loss -> discretize -> freeze ==="
                + (" [residual]" if cfg.group_residual else "")
                + (f" [recur={cfg.recur}]" if cfg.recur > 1 else "")
                + (f" [local={cfg.local}]"
                   if getattr(cfg, "local", 0) > 0 else "")
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
    def _local_wires(self, in_dim, d0, k):
        """--local K: 各ゲートに画素位置を割り当て、K×K近傍のプールビットから
        2入力を抽選する(畳み込み配線のPhase 1=局所性の事前分布のみ、重み共有なし)。
        ゲート位置はランダム割当 — GroupSumのクラス群(ゲート番号順)と位置の相関を
        切るため(等間隔割当だとクラス0が画像上部しか見えない)。warm併用時は
        前層のゲート位置を継承(恒等=前層コピーの意味論と整合)。
        近傍に候補が無ければ窓を1ずつ広げる(MNISTでG<npixのため)"""
        cfg = self.cfg
        w, npix, G = self._w, self._npix, cfg.gates
        g = torch.Generator().manual_seed(cfg.seed * 400 + d0 + k + 1)
        buckets = [[] for _ in range(npix)]
        for i, p in enumerate(self.pos_pool):
            buckets[p].append(i)
        if cfg.warm_start > 0 and (d0 + k) > 0:
            gpos = list(self.pos_pool[-G:])     # 前層ゲート位置を継承(プール末尾)
        else:
            gpos = torch.randint(0, npix, (G,), generator=g).tolist()
        ia, ib = [], []
        for gi in range(G):
            r0, c0 = divmod(gpos[gi], w)
            rad = cfg.local // 2
            cand = []
            while not cand:                     # 空窓なら拡張
                cand = [j for r in range(max(0, r0 - rad), min(w, r0 + rad + 1))
                        for c in range(max(0, c0 - rad), min(w, c0 + rad + 1))
                        for j in buckets[r * w + c]]
                rad += 1
            ia.append(cand[int(torch.randint(len(cand), (1,), generator=g))])
            ib.append(cand[int(torch.randint(len(cand), (1,), generator=g))])
        self._pending_pos = gpos
        return torch.tensor(ia), torch.tensor(ib)
    def make_layer(self, in_dim, d0, k):
        # 前層があるとき(d0+k>0)だけ恒等warm-start。最初の層は前層が無いので素のまま
        warm = self.cfg.warm_start if (d0 + k) > 0 else 0.0
        wires = (self._local_wires(in_dim, d0, k)
                 if getattr(self.cfg, "local", 0) > 0 else None)
        return LogicLayer(in_dim, self.cfg.gates,
                          seed=self.cfg.seed * 100 + d0 + k + 1, warm=warm,
                          wires=wires)
    def train(self, win, layers, d0):
        """窓内W層をsoftのまま共同学習(receding horizonの1ステップ)。
        損失は窓の最終層のCE(--win-loss last)か全層平均(--win-loss all)。
        W=1のとき従来のgreedy 1層学習と厳密に一致する"""
        cfg = self.cfg
        opt = torch.optim.Adam([p for L in win for p in L.parameters()], lr=cfg.lr)
        def loss(idx):
            # uint8常駐プールをこのバッチ分だけfloatへ(0/1なので厳密・ビット等価)
            pool = (self.pool_tr if idx is None else self.pool_tr[idx]).float()
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
                # --recur: 同じ層をsoftのまま反復適用(重み共有、勾配はK回展開を流れる)
                x = pool
                for _ in range(reps(pool.shape[1], cfg)):
                    h = L(x)
                    x = h
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
        h_tr, h_te = self.pool_tr, self.pool_te
        for _ in range(reps(self.pool_tr.shape[1], cfg)):   # --recur: hardでK回反復
            h_tr, h_te = hard_batched(L, h_tr), hard_batched(L, h_te)
        s_tr = group_sum(h_tr, cfg.n_class, self.tau)
        s_te = group_sum(h_te, cfg.n_class, self.tau)
        if cfg.group_residual:
            self.accum_tr = self.accum_tr + s_tr    # 累積更新(= 次層のオフセット)
            self.accum_te = self.accum_te + s_te
            s_tr, s_te = self.accum_tr, self.accum_te
        a_te = accuracy(s_te, self.yte)
        a_tr = accuracy(s_tr, self.ytr)
        # プールはuint8で前進(hardビットは厳密に0.0/1.0なので変換は可逆)
        self.pool_tr = next_pool(h_tr.to(torch.uint8), self.X_u8, self.pool_tr, cfg)
        self.pool_te = next_pool(h_te.to(torch.uint8), self.Xte_u8, self.pool_te, cfg)
        # --local: プール位置も同じ規則で前進(no-skip=ゲート位置のみ、
        # skip-input=[入力位置 || ゲート位置])
        if getattr(cfg, "local", 0) > 0:
            self.pos_pool = (self.pos_input + self._pending_pos
                             if cfg.skip_input else list(self._pending_pos))
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

