"""共有基盤: 16ゲートの実数緩和、LogicLayer、データ読み込み、学習ユーティリティ
(fit / 適応エポックの停止判定 / プール前進 / GroupSum集計)。
どの実験モジュールもここに依存する。挙動はexperiment.py単一ファイル時代と
ビット等価(tests.pyでピン留め)。"""
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
    def __init__(self, in_dim, n_gates, seed, struct=None, warm=0.0, wires=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        ia = torch.randint(0, in_dim, (n_gates,), generator=g)
        ib = torch.randint(0, in_dim, (n_gates,), generator=g)
        logits = torch.randn(n_gates, 16, generator=g)
        # wires=(ia, ib): 配線の外部指定(--local の局所配線など)。主ドローの後に
        # 上書きするので wires=None のとき従来とビット単位で一致する。
        # warm(恒等)より先に適用 — warm併用時は ia が恒等に再上書きされ ib が残る
        if wires is not None:
            ia, ib = wires[0].clone(), wires[1].clone()
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
def _parse_thresholds(spec, Xtrain_raw):
    """--thresholds の解釈。"5,10,15"=絶対閾値 / "q4"=train非ゼロ画素の等間隔
    分位点でK面(閾値増にも対応)。分位点はtrainのみから計算(テストリーク防止)"""
    if spec.startswith("q"):
        K = int(spec[1:])
        nz = Xtrain_raw[Xtrain_raw > 0]
        qs = [100.0 * k / (K + 1) for k in range(1, K + 1)]
        ths = sorted(set(float(np.percentile(nz, q)) for q in qs))
    else:
        ths = sorted(float(t) for t in spec.split(","))
    return ths

def load_data(dataset="digits", seed=0, thresholds=None):
    """thresholds=None は従来の固定サーモメータ(digits 3/7/11, mnist 63/127/191)
    とビット等価。指定時のみ新しい二値化パスを通る(タスク23: 入力二値化改善)"""
    if dataset == "mnist":
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="liac-arff")  # no pandas needed
        y = y.astype(np.int64)                   # 28x28 pixels, values 0..255
        ths = ((63, 127, 191) if thresholds is None
               else _parse_thresholds(thresholds, X[:60000]))
        Xb = np.concatenate([(X > t).astype(np.float32) for t in ths], axis=1)
        return (torch.tensor(Xb[:60000]), torch.tensor(Xb[60000:]),   # standard split
                torch.tensor(y[:60000]), torch.tensor(y[60000:]))
    X, y = load_digits(return_X_y=True)          # 8x8 digits, values 0..16
    if thresholds is None:                       # 従来パス(ビット等価)
        Xb = np.concatenate([(X > t).astype(np.float32) for t in (3, 7, 11)], axis=1)
        Xtr, Xte, ytr, yte = train_test_split(
            Xb, y, test_size=0.25, stratify=y, random_state=seed)
    else:
        # 分位点をtrainだけで計算するため先に生画素で分割(random_state・stratifyが
        # 同じなら分割インデックスは二値化前後で同一)
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed)
        ths = _parse_thresholds(thresholds, Xtr_raw)
        Xtr = np.concatenate([(Xtr_raw > t).astype(np.float32) for t in ths], axis=1)
        Xte = np.concatenate([(Xte_raw > t).astype(np.float32) for t in ths], axis=1)
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

def reps(pool_w, cfg):
    """--recur: この層を何回反復適用するか。重み共有の再帰は配線次元が
    合うとき(プール幅==gates、no-skipの層2以降)だけ成立。層1(入力幅)は1回。
    recur=1(既定)は常に1で従来とビット等価"""
    r = getattr(cfg, "recur", 1)
    return r if r > 1 and pool_w == cfg.gates else 1

def hard_pass(layers, X, cfg):
    """凍結済み層をhardで順に評価し、(最終層の出力h, 次層の配線プール)を返す。
    layersが空のときは (None, X)。--recur>1では各層をreps回反復適用"""
    pool, h = X, None
    for L in layers:
        x = pool
        for _ in range(reps(pool.shape[1], cfg)):
            h = hard_batched(L, x)
            x = h                    # 反復中は h がそのまま次の入力(no-skip前提)
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

