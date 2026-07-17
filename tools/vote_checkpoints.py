"""チェックポイント投票: 学習済みの凍結回路(.pt)を読み込んで投票アンサンブルを
測る。追加学習はゼロ — --checkpointが保存しているのは設計図(配線ia/ib+ゲート選択
logits)そのものなので、回路を再構築して推論を流すだけで済む。

メンバーは (.pt, 深さ) の組で指定する:
  別シード同士        = 本物の多様性(配線の抽選が違う独立回路)
  同一シードの深さ違い = prefix(入れ子)構造なのでメンバー間の相関が高い
                        (残差では深さd+kの予測 = 深さdの予測 + その先k層の補正)

残差readout対応: 予測は全層のクラス別ビットの累積和(src/groupsum.py の commit と
同じ規則)。既存の GroupSum.counts() は最終層のhしか読まないので残差には使えない
— ここで累積を組む。投票は整数カウントのまま合算する(τ除算後のfloatを
メンバー間で足すと同点クラスの丸め順序でargmaxが割れる = issue #8の教訓)。

使い方:
    python tools/vote_checkpoints.py --members runs/a.pt:41 runs/b.pt:78
    python tools/vote_checkpoints.py --members runs/a.pt:41 runs/a.pt:46 runs/a.pt:51
"""
import argparse, os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
import torch.nn.functional as F
import checkpoint
from core import (load_data, group_sum, accuracy, next_pool, hard_batched, reps)


def scores_by_depth(path, depths, device):
    """.ptを1回だけ再生し、指定された各深さでの累積クラススコアを返す。
    戻り値: ({深さ: [B, n_class] 整数カウント}, fingerprint, yte)"""
    fp, saved = checkpoint.load(path)
    cfg = SimpleNamespace(**fp)
    cfg.device = device
    if not cfg.group_residual:
        raise SystemExit(f"{path}: このツールは残差readout(--group-residual)専用")
    if getattr(cfg, "conv", 0):
        raise SystemExit(f"{path}: conv回路は未対応(プーリング付き累積が別物のため)")
    if max(depths) > len(saved):
        raise SystemExit(f"{path}: 深さ{max(depths)}は未確定(保存済みは{len(saved)}層)")
    _, Xte, _, yte = [t.to(device) for t in
                      load_data(cfg.dataset, thresholds=cfg.thresholds)]
    pool = Xte
    accum = torch.zeros(len(yte), cfg.n_class, device=device)
    out = {}
    for d, dd in enumerate(saved, 1):
        if d > max(depths):
            break
        L = checkpoint.rebuild(dd, device)
        h = pool
        for _ in range(reps(pool.shape[1], cfg)):   # --recur(task29は1回)
            h = hard_batched(L, h)
        accum = accum + group_sum(h, cfg.n_class, 1.0)   # τで割らない=厳密な整数
        if d in depths:
            out[d] = accum.clone()
        pool = next_pool(h, Xte, pool, cfg)
    return out, fp, yte


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", required=True, metavar="PATH:DEPTH",
                   help="投票メンバー。例 runs/seed1.pt:41 runs/seed2.pt:78")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    want = {}          # {path: set(深さ)} — 同じ.ptは1回の再生でまとめて測る
    order = []
    for m in a.members:
        path, _, d = m.rpartition(":")
        path, d = path or m, int(d)
        want.setdefault(path, set()).add(d)
        order.append((path, d))

    counts, yte, fps = {}, None, []
    for path, depths in want.items():
        sc, fp, y = scores_by_depth(path, sorted(depths), a.device)
        counts.update({(path, d): s for d, s in sc.items()})
        yte, _ = y, fps.append((path, fp))

    # 別シードのメンバーは同じデータ・同じ設定でなければ投票の意味がない
    keys = [k for k in fps[0][1] if k != "seed"]
    for path, fp in fps[1:]:
        diff = {k: (fps[0][1][k], fp[k]) for k in keys if fps[0][1][k] != fp[k]}
        if diff:
            raise SystemExit(f"設定がメンバー間で不一致 ({path}): {diff}")
    n_class = fps[0][1]["n_class"]

    print(f"=== checkpoint vote over {len(order)} members ===")
    for path, d in order:
        print(f"  {os.path.basename(path)} @depth {d:3d}"
              f"  (seed {dict(fps)[path]['seed']}) ="
              f" {accuracy(counts[(path, d)], yte):.4f}")
    C = torch.stack([counts[k] for k in order])              # [M, B, n_class]
    soft = accuracy(C.sum(0), yte)
    votes = F.one_hot(C.argmax(-1), n_class).sum(0).float()
    c = C.sum(0)
    t = (c - c.amin(1, keepdim=True)) / (c.amax(1, keepdim=True)
                                         - c.amin(1, keepdim=True) + 1e-9)
    maj = accuracy(votes + 0.99 * t, yte)                    # 同数タイは合算で決着
    mean = float(np.mean([accuracy(counts[k], yte) for k in order]))
    # メンバー同士がどれだけ違う答えを出すか(多様性=アンサンブルの燃料)
    preds = C.argmax(-1)
    agree = float((preds == preds[0]).all(0).float().mean())
    print(f"  member mean                        = {mean:.4f}")
    print(f"  all members agree on               = {agree:.1%} of samples")
    print(f"  soft vote (summed GroupSum counts) = {soft:.4f}")
    print(f"  majority vote (count tie-break)    = {maj:.4f}")


if __name__ == "__main__":
    main()
