"""オフトラックのスケーリングレバー(SCALING.md対応): アンサンブル投票。
固定予算の主線(greedy単発)とは別扱い — 効くと分かっている資源投入。"""
import numpy as np
import torch
import torch.nn.functional as F
from core import accuracy
from greedy import run_greedy, make_objective

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

