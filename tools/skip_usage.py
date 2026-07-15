"""skip線の深さ別利用度診断(2026-07-15)。experiment.pyは変更しない。
問い:「skip線(生入力X)の効果は層が進むと弱くなるのか?」
配線ia/ibは固定ランダムで学習されない(core.LogicLayer)ので、学習の自由度は
各ゲートの関数(fn)選択のみ。そこで残差+skip構成で層ごとに
  usage_X = fnが実際に読む入力スロット数 / X線(idx<in_bits)スロット総数
  usage_h = 同、前層h線
を数える(dead=テスト集合で定数出力のゲートはfn選択が無意味なので除外)。
skipの効果が深さで弱まるなら usage_X が層とともに下がるはず。
結果(digits, seed1/2, 16層): usage_Xは70-82%で横ばい・単調減衰なし
(HISTORY.md 2026-07-15エントリ参照)。これは「利用度」であって
精度への寄与(ablation)ではない点に注意。
使い方: python tools/skip_usage.py [--seed 1] [--max-layers 16] [--csv out.csv]"""
import sys, os, argparse, types, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))  # 実装はsrc/
import numpy as np
import torch
from core import load_data
from groupsum import GroupSum

# fnが入力aを読まないゲート / bを読まないゲート(GATE_NAMES順のindex)
NO_A = {0, 5, 10, 15}   # FALSE, B, !B, TRUE
NO_B = {0, 3, 12, 15}   # FALSE, A, !A, TRUE

def make_cfg(**kw):
    d = dict(gates=500, n_class=10, seed=1, lr=0.05, epochs=120,
             device="cuda" if torch.cuda.is_available() else "cpu", batch=0,
             group_residual=True, group_boost=1.0, group_loss="ce", warm_start=0.0,
             window=1, commit=1, win_loss="all", skip_input=True, skip_all=False,
             recur=1, epoch_stop=0.0, epoch_peak=0.0, epoch_peak_decay=1.0,
             epoch_chain=0.0, epoch_min=70, epoch_check=5, epoch_patience=3,
             dataset="digits")
    d.update(kw)
    return types.SimpleNamespace(**d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gates", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--max-layers", type=int, default=16)
    ap.add_argument("--csv", default=None, help="数値の裏取り用CSV出力先(省略時なし)")
    a = ap.parse_args()
    cfg = make_cfg(seed=a.seed, gates=a.gates, epochs=a.epochs)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    Xtr, Xte, ytr, yte = [t.to(cfg.device) for t in load_data("digits", seed=cfg.seed)]
    in_bits = Xtr.shape[1]
    obj = GroupSum(Xtr, Xte, ytr, yte, cfg)
    rows = []
    print(f"=== skip-usage by depth (digits, residual+skip, {cfg.gates} gates, "
          f"seed {cfg.seed}, in_bits={in_bits}) ===")
    print("layer  usage_X%  usage_h%  ratio  slotsX  dead%  probe_te")
    layers = []
    for d0 in range(a.max_layers):
        L = obj.make_layer(obj.begin(layers, d0), d0, 0).to(cfg.device)
        obj.train([L], layers, d0)
        layers.append(L)
        a_tr, a_te = obj.commit(layers, L)
        # 層1はプール=Xのみ(X/hの区別がない)なので集計から除外
        if d0 == 0:
            print(f"{d0+1:5d}  (layer1: pool=X only)                    {a_te:.4f}")
            continue
        fn = L.logits.argmax(-1).cpu()
        ia, ib = L.ia.cpu(), L.ib.cpu()
        h = obj.pool_te[:, in_bits:].float()   # skipプール=[X, h] の末尾gates本=この層
        s = h.sum(0)
        alive = ((s > 0) & (s < len(h))).cpu()
        usedX = usedH = slotX = slotH = 0
        for g in range(cfg.gates):
            for idx, dead_set in ((int(ia[g]), NO_A), (int(ib[g]), NO_B)):
                is_x = idx < in_bits
                if is_x: slotX += 1
                else:    slotH += 1
                if not bool(alive[g]):
                    continue
                if int(fn[g]) not in dead_set:   # fnがこのスロットを実際に読む
                    if is_x: usedX += 1
                    else:    usedH += 1
        uX = usedX / max(slotX, 1); uH = usedH / max(slotH, 1)
        dead_pct = 100 * (1 - alive.float().mean().item())
        rows.append(dict(layer=d0 + 1, usage_X=uX, usage_h=uH,
                         ratio=uX / uH if uH > 0 else float("nan"),
                         slotsX=slotX, dead_pct=dead_pct, probe_te=a_te))
        print(f"{d0+1:5d}  {uX*100:7.1f}  {uH*100:7.1f}  {uX/uH if uH else 0:5.2f}"
              f"  {slotX:6d}  {dead_pct:5.1f}  {a_te:.4f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"csv -> {a.csv}")

if __name__ == "__main__":
    main()
