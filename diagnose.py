"""回路診断(学習不要寄り・CLAUDE.mdタスク24)。experiment.pyは変更しない。
(a) 学習が触らなかったゲート率: 各層で学習前後のargmax(選択されたゲート)が
    変わらなかった割合。ランダム初期argmaxのまま学習が動かさなかったゲート。
(b) 機能的冗長度: hard回路で各ゲートの出力ビット列(テスト集合上)を見て、
    定数列(dead)・他ゲートと同一/相補(complement)な列の割合を数える。
W=1の素のgreedyループを再現(--group-residual等のモードは付けない素の診断)。"""
import sys, argparse, types
import numpy as np
import torch
from core import load_data, group_sum, accuracy, hard_batched, next_pool, GATE_NAMES
from groupsum import GroupSum

def make_cfg(**kw):
    d = dict(gates=500, n_class=10, seed=1, lr=0.05, epochs=120,
             device="cuda" if torch.cuda.is_available() else "cpu", batch=0,
             group_residual=False, group_boost=1.0, group_loss="ce", warm_start=0.0,
             window=1, commit=1, win_loss="all", skip_input=False, skip_all=False,
             recur=1, epoch_stop=0.0, epoch_peak=0.0, epoch_peak_decay=1.0,
             epoch_chain=0.0, epoch_min=70, epoch_check=5, epoch_patience=3,
             dataset="digits")
    d.update(kw)
    return types.SimpleNamespace(**d)

def col_signature(bits):
    """[B]の0/1列をbytesに(ハッシュ用)。相補はNOTして正規化した鍵と別に持つ"""
    b = bits.to(torch.uint8).cpu().numpy().tobytes()
    return b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--max-layers", type=int, default=5)
    ap.add_argument("--warm-start", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = make_cfg(gates=a.gates, epochs=a.epochs, warm_start=a.warm_start, seed=a.seed)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in load_data("digits")]
    obj = GroupSum(Xtr, Xte, ytr, yte, cfg)

    print(f"=== circuit diagnostics (digits, {cfg.gates} gates, warm={cfg.warm_start}, "
          f"seed {cfg.seed}, device {cfg.device}) ===")
    print("layer  untouched%  dead%  dup/compl%  distinct-fn  probe_test")
    layers, all_cols, seen = [], {}, {}
    gate_hist = np.zeros(16, dtype=int)   # 学習後に選ばれたゲート種類の集計
    for d0 in range(a.max_layers):
        L = obj.make_layer(obj.begin(layers, d0), d0, 0).to(cfg.device)
        init = L.logits.argmax(-1).clone()          # (a) 学習前の選択
        obj.train([L], layers, d0)
        final = L.logits.argmax(-1)                 # (a) 学習後の選択
        gate_hist += np.bincount(final.cpu().numpy(), minlength=16)
        untouched = (init == final).float().mean().item()
        layers.append(L)
        a_tr, a_te = obj.commit(layers, L)          # 凍結してプール前進

        # (b) この層のhard出力列をテスト集合上で評価(commit後のpool_teは次層入力=
        #     この層の出力なので、直前のpoolから作り直す)
        pool_te = obj.pool_te                        # commitで前進済み = この層の出力
        # pool_teはno-skip/skipでレイアウトが違うが、末尾gates本がこの層の出力
        h = pool_te[:, -cfg.gates:]
        dead = 0; dup = 0
        for j in range(cfg.gates):
            colj = h[:, j]
            s = colj.sum().item()
            if s == 0 or s == len(colj):             # 定数列 = dead
                dead += 1; continue
            key = col_signature(colj)
            keyc = col_signature(1 - colj)           # 相補
            if key in seen or keyc in seen:
                dup += 1
            else:
                seen[key] = (d0, j)
        distinct = cfg.gates - dead - dup
        print(f"{len(layers):5d}  {untouched*100:8.1f}  {dead/cfg.gates*100:5.1f}"
              f"  {dup/cfg.gates*100:9.1f}  {distinct:11d}  {a_te:.4f}")

    # (c) 学習後に選ばれたゲート種類の分布(全層合計)。difflogic系の偏り知見と比較
    tot = gate_hist.sum()
    print(f"\ngate-type histogram over all {tot} gates (chosen after training):")
    for i in np.argsort(gate_hist)[::-1]:
        if gate_hist[i] == 0:
            continue
        bar = "#" * int(gate_hist[i] * 40 / gate_hist.max())
        print(f"  {GATE_NAMES[i]:5s} {gate_hist[i]*100/tot:5.1f}%  {bar}")

if __name__ == "__main__":
    main()
