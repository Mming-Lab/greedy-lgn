"""学習済みseqセルの力学系診断(発振器・レジスタ探し)。
「再帰はフリップフロップだけでなく発振器も作れる」(プロジェクトオーナーの仮説)
を検証する: --seqで学習した再帰セルに一定入力(空行=ゼロ)を与えて自励
ロールアウトし、状態が入る周期軌道を検出する。
  周期1(固定点)  = レジスタ的(値を保持)
  周期2以上      = 発振器(クロック/カウンタ/LFSR的)
hardビットの有限状態機械なので軌道は必ず最終的に周期的。問いは
「学習がどんな周期を作ったか」と「データ由来の状態から何ステップで安定するか」。
experiment.py非改変の独立診断。"""
import argparse, types
import numpy as np
import torch
from core import load_data, hard_batched
from seq import SeqGroupSum

def make_cfg(**kw):
    d = dict(gates=500, n_class=10, seed=1, lr=0.05, epochs=120,
             device="cuda" if torch.cuda.is_available() else "cpu", batch=0,
             group_residual=False, group_boost=1.0, group_loss="ce",
             warm_start=3.0, window=1, commit=1, win_loss="all",
             skip_input=False, skip_all=False, recur=1, seq=True,
             epoch_stop=0.0, epoch_peak=0.0, epoch_peak_decay=1.0,
             epoch_chain=0.0, epoch_min=70, epoch_check=5, epoch_patience=3,
             dataset="digits")
    d.update(kw)
    return types.SimpleNamespace(**d)

@torch.no_grad()
def cycle_census(L, x_const, s0, max_steps=512):
    """一定入力x_constの下でs0からロールアウトし、(遷移到達步数, 周期長)を返す。
    状態ハッシュの初回再訪で検出(有限状態機械なので必ず見つかるか打ち切り)"""
    seen = {}
    s = s0.view(1, -1).clone()          # [1, G] に統一
    x = x_const.view(1, -1)             # [1, row_bits]
    traj = []
    for t in range(max_steps):
        key = s.to(torch.uint8).cpu().numpy().tobytes()
        if key in seen:
            mu = seen[key]              # 過渡の長さ
            lam = t - mu                # 周期
            return mu, lam, traj[mu:t]  # 周期軌道の状態列
        seen[key] = t
        traj.append(s[0].clone())
        s = hard_batched(L, torch.cat([x, s], 1))
    return None, None, None             # 打ち切り(周期>max_steps)

def bit_periods(cycle):
    """周期軌道内の各状態ビットの最小周期を数える(1=定数/レジスタ, >1=発振)"""
    A = torch.stack(cycle).to(torch.uint8).cpu().numpy()   # [lam, G]
    lam, G = A.shape
    per = np.zeros(G, dtype=int)
    for j in range(G):
        col = A[:, j]
        for p in range(1, lam + 1):
            if lam % p == 0 and (col == np.roll(col, p)).all():
                per[j] = p
                break
    return per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--max-layers", type=int, default=3)
    ap.add_argument("--n-init", type=int, default=16, help="ランダム初期状態の本数")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = make_cfg(warm_start=a.warm_start, epochs=a.epochs, seed=a.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in load_data("digits")]
    obj = SeqGroupSum(Xtr, Xte, ytr, yte, cfg)

    print(f"=== seq cell dynamics (digits, {cfg.gates} gates, warm={cfg.warm_start},"
          f" T={obj.T}, seed {cfg.seed}) ===")
    layers = []
    g = torch.Generator().manual_seed(999)
    for d0 in range(a.max_layers):
        L = obj.make_layer(obj.begin(layers, d0), d0, 0).to(cfg.device)
        obj.train([L], layers, d0)
        layers.append(L)
        a_tr, a_te = obj.commit(layers, L)
        # 自励ロールアウト: 入力=空行(ゼロ)。初期状態はゼロ+ランダムn本
        x0 = torch.zeros(obj.seq_tr[0].shape[1], device=cfg.device)
        inits = [torch.zeros(cfg.gates, device=cfg.device)]
        inits += [(torch.rand(cfg.gates, generator=g) > 0.5).float().to(cfg.device)
                  for _ in range(a.n_init)]
        stats, fixed_pts, transients = {}, set(), []
        osc_example = None
        for s0 in inits:
            mu, lam, cyc = cycle_census(L, x0, s0)
            if lam is None:
                stats[">512"] = stats.get(">512", 0) + 1
                continue
            stats[lam] = stats.get(lam, 0) + 1
            transients.append(mu)
            if lam == 1:            # 固定点の同一性(多安定=レジスタ / 単一=忘却)
                fixed_pts.add(cyc[0].to(torch.uint8).cpu().numpy().tobytes())
            elif osc_example is None:
                osc_example = (mu, lam, bit_periods(cyc))
        summary = ", ".join(f"period {k}: {v}" for k, v in sorted(
            stats.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])))
        print(f"layer {d0+1}: probe test={a_te:.4f}  attractors({len(inits)} inits)"
              f" -> {summary}")
        if fixed_pts:
            print(f"    distinct fixed points: {len(fixed_pts)}"
                  f"  transient steps: min={min(transients)}"
                  f" mean={sum(transients)/len(transients):.1f}"
                  f" max={max(transients)}  (task horizon T={obj.T})")
        if osc_example:
            mu, lam, per = osc_example
            n_osc = int((per > 1).sum())
            top = sorted(set(per.tolist()) - {1})
            print(f"    oscillator found: transient={mu}, cycle={lam},"
                  f" oscillating bits={n_osc}/{cfg.gates} (bit periods {top})")

if __name__ == "__main__":
    main()
