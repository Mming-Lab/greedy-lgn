"""Forward-Forward目的: goodness(popcount)を正例で上げ負例で下げる。
ラベルは入力に重畳、推論は全ラベル試行。負例マイニング(--ff-neg系)と
構造化配線(--ff-struct)を含む。"""
import numpy as np
import torch
import torch.nn.functional as F
from core import LogicLayer, accuracy, next_pool, hard_pass, fit

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

