"""Regression tests: pinned, bit-exact results for published configurations.

このリポジトリの生命線は「公開済みの数字がビット単位で再現できること」。
experiment.py をリファクタ・機能追加するときは、必ずこのスイートを通すこと
(数値は丸めではなく厳密一致で比較する。1e-4 の丸め後の値が変わったら、
RNGの消費順序か演算順序が変わっている)。

Usage:
    python tests.py                # CPU, ~10-15 min
    python tests.py --device cuda  # GPU, ~2 min (numbers are identical to CPU)
    python tests.py --only ff      # substring filter on case names
"""
import argparse, json, subprocess, sys, time

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
]


def run_case(name, args, expect, device):
    cmd = [sys.executable, "experiment.py"] + args.split() + ["--device", device]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if out.returncode != 0:
        return False, dt, f"exit {out.returncode}\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    try:
        summary = json.loads(out.stdout.split("=== summary ===", 1)[1])
    except (IndexError, json.JSONDecodeError) as e:
        return False, dt, f"summary parse failed: {e}\n{out.stdout[-2000:]}"
    bad = [f"  {k}: expected {v!r}, got {summary.get(k)!r}"
           for k, v in expect.items() if summary.get(k) != v]
    return (not bad), dt, "\n".join(bad)


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
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
