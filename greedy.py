"""手法の本体: greedy layer-wise ループ(目的関数非依存の骨格)。
窓Wをsoftで学習→先頭Jを離散化・凍結→スライド。プローブ頭打ちで深さ確定。"""
import time
from groupsum import GroupSum
from ff import ForwardForward

def make_objective(Xtr, Xte, ytr, yte, cfg):
    cls = ForwardForward if cfg.objective == "ff" else GroupSum
    return cls(Xtr, Xte, ytr, yte, cfg)

# ----------------------------- (A) greedy layer-wise -----------------------------
def run_greedy(Xtr, Xte, ytr, yte, cfg):
    """greedy本体(目的関数に依存しない骨格): W層の窓をsoftで学習し、先頭J層を
    離散化・凍結して窓をスライド。プローブが頭打ちになったら深さを確定する"""
    obj = make_objective(Xtr, Xte, ytr, yte, cfg)
    W, J = cfg.window, cfg.commit
    print(obj.header()
          + (" [carry]" if cfg.carry else "")
          + (" [skip-all wiring]" if cfg.skip_all else
             " [skip-input wiring]" if cfg.skip_input else ""))
    layers, best_acc, best_depth, since_best = [], -1.0, 0, 0
    t0 = time.time()
    stop = False
    carry = []   # --carry: コミットされなかった先読み層を次スライドへ受け継ぐ足場方式
    while not stop and len(layers) < cfg.max_layers:
        d0 = len(layers)
        # 凍結済みプレフィックスの上にW層の窓を新規作成(スライドごとに再計画。
        # コミットされなかったlookahead層は捨てる = receding horizon)
        win, in_dim = [], obj.begin(layers, d0)
        for k in range(W):
            if k < len(carry):
                win.append(carry[k])   # 受け継いだ層(重み保持、no-skipでin_dim=G一致)
            else:
                win.append(obj.make_layer(in_dim, d0, k).to(cfg.device))
            in_dim = (in_dim + cfg.gates if cfg.skip_all else
                      obj.X.shape[1] + cfg.gates if cfg.skip_input else cfg.gates)
        obj.train(win, layers, d0)
        carry = win[J:] if cfg.carry else []
        for L in win[:J]:  # 窓の先頭J層だけ離散化して凍結(HARDビット上で確定)
            layers.append(L)
            a_tr, a_te = obj.commit(layers, L)
            print(f"  layer {len(layers)}: {obj.kind} probe"
                  f"  train={a_tr:.4f}  test={a_te:.4f}")
            if a_te > best_acc + 1e-4:
                best_acc, best_depth, since_best = a_te, len(layers), 0
            else:
                since_best += 1
                if since_best >= cfg.patience:
                    print(f"  -> stop: no improvement for {cfg.patience} layers")
                    stop = True
                    break
            if len(layers) >= cfg.max_layers:
                break
    print(f"  {obj.tag}: best {obj.kind} test acc = {best_acc:.4f}"
          f" at depth {best_depth}  ({time.time() - t0:.0f}s)\n")
    return layers[:best_depth], best_acc, best_depth

