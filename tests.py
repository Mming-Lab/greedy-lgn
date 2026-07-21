"""Regression tests: pinned, bit-exact results for published configurations.

このリポジトリの生命線は「公開済みの数字がビット単位で再現できること」。
experiment.py と各モジュール(core/groupsum/ff/greedy/scaling/e2e/simplify)を
リファクタ・機能追加するときは、必ずこのスイートを通すこと
(数値は丸めではなく厳密一致で比較する。1e-4 の丸め後の値が変わったら、
RNGの消費順序か演算順序が変わっている)。

Usage:
    python tests.py                # CPU, ~10-15 min
    python tests.py --device cuda  # GPU, ~4 min
    python tests.py --only ff      # substring filter on case names

CPUとCUDAはほぼ全ケースでビット等価だが、conv だけは縮約順序の実装差で
1サンプル割れる。期待値は {"cpu": ..., "cuda": ...} と書けば実行デバイスの
値が使われる(丸め許容は入れない — 検出力を落とさないため)。
"""
import argparse, json, os, subprocess, sys, tempfile, time

# (name, experiment.py args, expected summary fields)
# 期待値はすべて実測でピン留めした公開値。seed 1 / digits。
CASES = [
    ("smoke groupsum",
     "--gates 200 --epochs 30 --max-layers 3 --skip-e2e",
     {"greedy_hard_test_acc": 0.6067, "greedy_depth": 2}),
    ("default greedy (v0.1 headline)",
     "--skip-e2e",
     {"greedy_hard_test_acc": 0.8822, "greedy_depth": 4,
      "gates_before": 2000, "gates_after_simplify": 1316}),
    ("windowed lookahead W2J2",
     "--skip-e2e --window 2 --commit 2 --win-loss all",
     {"greedy_hard_test_acc": 0.9044, "greedy_depth": 4}),
    # soft voteの期待値は0.9111(整数カウント投票の決定的な値)。当初issue #8で
    # 公開した0.9133は、τ除算後のfloatをメンバー間で合算したときに同点クラスの
    # 丸め順序が偶然転んだGPU固有の値だった(このスイートが発見した不一致)。
    ("ensemble x4 voting",
     "--skip-e2e --ensemble 4",
     {"member_hard_test_acc": [0.8822, 0.88, 0.8889, 0.8578],
      "depths": [4, 2, 3, 2],
      "ensemble_soft_vote_acc": 0.9111,
      "ensemble_majority_vote_acc": 0.9022}),
    ("ff rep38",
     "--skip-e2e --objective ff --ff-label-rep 38",
     {"greedy_hard_test_acc": 0.86, "greedy_depth": 6}),
    ("ff review-warmup W2J2 (best fixed-budget stack)",
     "--skip-e2e --objective ff --ff-label-rep 38 --window 2 --commit 2"
     " --ff-neg review --ff-neg-warmup 0.5",
     {"greedy_hard_test_acc": 0.9022, "greedy_depth": 6}),
    # 2026-07-13 追加分。高速なgates 200予算でコードパスをピン(値自体は小さい
    # 設定のもので、READMEの500ゲートの数字とは別物 — 目的は回帰の壊れ検出)。
    ("warm-start identity init",
     "--gates 200 --epochs 30 --max-layers 4 --warm-start 5 --skip-e2e",
     {"greedy_hard_test_acc": 0.6511, "greedy_depth": 4}),
    ("recur 2 (within-layer iteration)",
     "--gates 200 --epochs 30 --max-layers 4 --recur 2 --skip-e2e",
     {"greedy_hard_test_acc": 0.6022, "greedy_depth": 1}),
    ("residual + group-boost",
     "--gates 200 --epochs 30 --max-layers 4 --group-residual --group-boost 2 --skip-e2e",
     {"greedy_hard_test_acc": 0.78, "greedy_depth": 4,
      # residualのsimplify(全層出力対応、2026-07-13)のピン。等価assertも走る
      "gates_before": 800, "gates_after_simplify": 639}),
    ("seq row-sequential + warm-start",
     "--gates 200 --epochs 30 --max-layers 4 --seq --warm-start 3 --skip-e2e",
     {"greedy_hard_test_acc": 0.5956, "greedy_depth": 4}),
    # 2026-07-14 追加: convの初のピン(メモリ最適化=uint8プール+チャンネル
    # チャンクの導入時にmainで採取)。digitsはCc>=Cの1チャンク経路=従来と
    # ビット等価であることをピンで保証する(マルチチャンクのlogits勾配は
    # 縮約順序の丸めが変わりうるため対象外 — conv.py forwardのコメント参照)。
    # 2026-07-19: このケースだけCPUとCUDAで値が割れる(450中1サンプル、
    # 0.64 vs 0.6422)。採取時はCUDAだけを見ており、既定のCPU実行では最初から
    # 落ちていた(今日のcarry作業とは無関係の既存不一致 — コミット済みHEADでも
    # 再現することを確認済み)。原因は畳み込みの縮約順序の実装差で、これは
    # 直せない類のもの(docs/RESULTS.md「Numerical footnote」節)。丸め許容を
    # 入れると回帰検出力が落ちるので、両デバイスの値を厳密にピンする
    ("conv C64/tree3 + residual",
     "--conv 64 --conv-tree 3 --epochs 60 --max-layers 3 --group-residual"
     " --skip-e2e",
     {"greedy_hard_test_acc": {"cpu": 0.64, "cuda": 0.6422},
      "greedy_depth": 3}),
    # 2026-07-19 追加: --carry(コミットされなかった先読み層を次スライドへ持ち越す
    # 足場方式)。それまでスイートに1本も無かった。carryはwin[J:]を重み付きで
    # 次スライドのwin先頭へ差し込むので、窓の張り直し・配線indexの読み替えが
    # 絡む。skip版は「持ち越しで位置J+i→iに移っても読むプール内容が変わらない」
    # ことの回帰(2026-07-19にno-skip限定の制約を解除した経路)
    ("carry W4J1 + residual (no-skip)",
     "--gates 200 --epochs 30 --max-layers 4 --group-residual"
     " --window 4 --commit 1 --carry --skip-e2e",
     {"greedy_hard_test_acc": 0.8311, "greedy_depth": 4,
      "gates_before": 800, "gates_after_simplify": 603}),
    ("carry W4J1 + residual + skip-input",
     "--gates 200 --epochs 30 --max-layers 4 --group-residual --skip-input"
     " --window 4 --commit 1 --carry --skip-e2e",
     {"greedy_hard_test_acc": 0.8756, "greedy_depth": 4,
      "gates_before": 800, "gates_after_simplify": 635}),
]


def _run(args, device):
    """experiment.pyを1回サブプロセス起動してsummary dictを返す((summary, dt, err)、
    失敗時はsummary=None+errに詳細)。run_caseとrun_checkpoint_caseの共通土台"""
    cmd = [sys.executable, "experiment.py"] + args.split() + ["--device", device]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if out.returncode != 0:
        return None, dt, f"exit {out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    try:
        summary = json.loads(out.stdout.split("=== summary ===", 1)[1])
    except (IndexError, json.JSONDecodeError) as e:
        return None, dt, f"summary parse failed: {e}\n{out.stdout[-2000:]}"
    return summary, dt, None


def run_case(name, args, expect, device):
    summary, dt, err = _run(args, device)
    if summary is None:
        return False, dt, err
    # 期待値が {"cpu": ..., "cuda": ...} のdictなら実行デバイスの値を採る。
    # 大半のケースはCPU/CUDAでビット等価だが、convだけは縮約順序の実装差で
    # 1サンプルずれる(CASESのconv注記参照)。どちらのデバイスでも厳密一致を
    # 保ちたいので、丸め許容を入れずに両方をピンする
    exp = {k: (v[device] if isinstance(v, dict) else v)
           for k, v in expect.items()}
    bad = [f"  {k}: expected {v!r}, got {summary.get(k)!r}"
           for k, v in exp.items() if summary.get(k) != v]
    return (not bad), dt, "\n".join(bad)


def run_checkpoint_case(device,
                        base=("--gates 200 --epochs 30 --group-residual"
                              " --skip-input --skip-e2e --seed 1"),
                        stop_at=3, full_at=6):
    """--checkpointの往復(分割実行)がノンストップ実行とビット単位で一致するか。
    層を確定するたびにtorch.saveし、再開時にfingerprintを照合して復元する
    本番コードパスをそのまま通す(タスク29のMNIST一晩バッチ分割を支える機能)。
    max-layers stop_atで一度止め、同じ--checkpointパスでmax-layers full_atを
    再実行して最終summaryがノンストップ実行と厳密一致することを見る。
    baseの差し替えでconv版(ConvLogicLayerのleaf/logits/幾何の復元)も同じ枠で検証"""
    t0 = time.time()
    full, _, err1 = _run(f"{base} --max-layers {full_at}", device)
    if full is None:
        return False, time.time() - t0, f"baseline run failed:\n{err1}"
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ck.pt")
        part1, _, err2 = _run(f"{base} --max-layers {stop_at} --checkpoint {ck}",
                              device)
        if part1 is None:
            return False, time.time() - t0, f"part1 run failed:\n{err2}"
        part2, _, err3 = _run(f"{base} --max-layers {full_at} --checkpoint {ck}",
                              device)
        if part2 is None:
            return False, time.time() - t0, f"part2 (resume) run failed:\n{err3}"
    bad = [f"  {k}: baseline {full.get(k)!r} != resumed {part2.get(k)!r}"
           for k in full if full.get(k) != part2.get(k)]
    return (not bad), time.time() - t0, "\n".join(bad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", default="", help="run only cases whose name contains this")
    a = p.parse_args()
    cases = [c for c in CASES if a.only in c[0]]
    failed = 0
    for name, args, expect in cases:
        ok, dt, detail = run_case(name, args, expect, a.device)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  ({dt:.0f}s)")
        if not ok:
            print(detail)
            failed += 1
    total = len(cases)
    if a.only == "" or "checkpoint" in a.only:
        total += 1
        ok, dt, detail = run_checkpoint_case(a.device)
        print(f"[{'PASS' if ok else 'FAIL'}] checkpoint resume == non-stop  ({dt:.0f}s)")
        if not ok:
            print(detail)
            failed += 1
        # conv版: ConvLogicLayer(leaf/logits/幾何)の保存・復元と、リプレイ中の
        # 特徴マップ形状の前進(commitがL.C/L.Hp/L.Wpから取る)を検証
        total += 1
        ok, dt, detail = run_checkpoint_case(
            a.device, base=("--conv 16 --conv-tree 2 --epochs 30"
                            " --group-residual --skip-e2e --seed 1"),
            stop_at=1, full_at=3)
        print(f"[{'PASS' if ok else 'FAIL'}] conv checkpoint resume == non-stop"
              f"  ({dt:.0f}s)")
        if not ok:
            print(detail)
            failed += 1
        # carry版(2026-07-19): 凍結層に加えて「未コミットの先読み層」も保存・復元
        # できているかを見る。ここが欠けると再開後の窓先頭が新規層に化けるので、
        # 層stop_at+1以降がノンストップ実行とずれる = 最終summaryの不一致で落ちる。
        # no-skip と skip-input の両方(次元の詰め方が違うため別経路)
        for tag, extra in (("no-skip", ""), ("skip-input", " --skip-input")):
            total += 1
            ok, dt, detail = run_checkpoint_case(
                a.device, base=("--gates 200 --epochs 30 --group-residual"
                                " --window 4 --commit 1 --carry --skip-e2e"
                                " --seed 1" + extra),
                stop_at=2, full_at=4)
            print(f"[{'PASS' if ok else 'FAIL'}] carry checkpoint resume =="
                  f" non-stop ({tag})  ({dt:.0f}s)")
            if not ok:
                print(detail)
                failed += 1
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
