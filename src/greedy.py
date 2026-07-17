"""手法の本体: greedy layer-wise ループ(目的関数非依存の骨格)。
窓Wをsoftで学習→先頭Jを離散化・凍結→スライド。プローブ頭打ちで深さ確定。
--checkpoint/--stop-fileによる中断・再開はcheckpoint.py(groupsum専用、
experiment.py側のp.errorでff/seq/conv/ensemble/carryとは排他)。"""
import os, time
import checkpoint
from groupsum import GroupSum
from ff import ForwardForward
from seq import SeqGroupSum
from conv import ConvGroupSum

def make_objective(Xtr, Xte, ytr, yte, cfg):
    cls = (ConvGroupSum if getattr(cfg, "conv", 0) > 0
           else SeqGroupSum if getattr(cfg, "seq", False)
           else ForwardForward if cfg.objective == "ff" else GroupSum)
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
    stop = False

    def track(a_tr, a_te):
        # コミット直後の共通処理(通常運転・再開リプレイの両方から呼ぶ =
        # best_acc/best_depth/since_bestの更新規則を1箇所に保つ)
        nonlocal best_acc, best_depth, since_best, stop
        print(f"  layer {len(layers)}: {obj.kind} probe"
              f"  train={a_tr:.4f}  test={a_te:.4f}")
        if a_te > best_acc + 1e-4:
            best_acc, best_depth, since_best = a_te, len(layers), 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                print(f"  -> stop: no improvement for {cfg.patience} layers")
                stop = True

    # --checkpoint: 既存ファイルがあれば凍結済み層を復元して続きから再開。
    # commit()はRNG不使用の純関数(pool/accum/pos_poolはlayersだけから決まる)
    # なので、保存済み層を先頭から1つずつ「再生」すれば内部状態もbest_acc等も
    # 中断なし実行とビット単位で一致する状態に戻る(新規に降臨するのはこの後)
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        fp, saved = checkpoint.load(cfg.checkpoint)
        want = checkpoint.fingerprint(cfg)
        if fp != want:
            diff = {k: (fp.get(k), want.get(k))
                    for k in want if fp.get(k) != want.get(k)}
            raise SystemExit("--checkpoint fingerprint mismatch, refusing to"
                              f" resume (settings changed): {diff}")
        print(f"=== resumed from checkpoint: {len(saved)} layers ===")
        for d in saved:
            L = checkpoint.rebuild(d, cfg.device)
            layers.append(L)
            track(*obj.commit(layers, L))

    t0 = time.time()
    carry = []   # --carry: コミットされなかった先読み層を次スライドへ受け継ぐ足場方式
    while not stop and len(layers) < cfg.max_layers:
        # --stop-file: 次の層を確定した直後にきれいに終了するための合図。
        # 起動直後(layers空)は無視する = 「進行中の層を終えてから止まる」の
        # 文字通りの意味にし、layers=[]のままsimplify()に渡る空回路の穴を避ける
        if cfg.stop_file and len(layers) > 0 and os.path.exists(cfg.stop_file):
            print(f"  -> stop: stop-file found ({cfg.stop_file})")
            break
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
            track(*obj.commit(layers, L))
            if cfg.checkpoint:
                checkpoint.save(cfg.checkpoint, layers, cfg)
            if stop or len(layers) >= cfg.max_layers:
                break
    print(f"  {obj.tag}: best {obj.kind} test acc = {best_acc:.4f}"
          f" at depth {best_depth}  ({time.time() - t0:.0f}s)\n")
    return layers[:best_depth], best_acc, best_depth

