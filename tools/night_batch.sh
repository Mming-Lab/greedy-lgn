#!/usr/bin/env bash
# 夜間バッチ(直列)。各ランは --checkpoint で層コミット単位に保存 = 途中で
# 切れても翌回に続きから再開できる。停止方法(ユーザー):
#   touch runs/STOP_night
# を作ると、進行中のランは「今の層を確定してから」きれいに止まり(層を無駄に
# しない)、キュー全体もそこで終了する。
# 順番 = 価値順: (1)(2) carry+skip seed2/3 で95.76%を3シード化(最優先)、
# (3) conv funnel(別軸・ランタイム未知なので4hキャップ)、(4) thresholds(保険)。
set -u
cd /c/work/github_Mming-Lab/greedy-lgn
PY=.venv/Scripts/python.exe
STOP=runs/STOP_night
BLOG=runs/night_batch.log
rm -f "$STOP"
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$BLOG"; }

run(){  # $1=name  $2=timeout(空=なし)  残り=experiment.py引数
  local name=$1 tmo=$2; shift 2
  if [ -f "$STOP" ]; then log "HALT: user stop before $name"; exit 0; fi
  local ck="runs/night_$name.pt" lg="runs/night_$name.log"
  log "START $name (timeout=${tmo:-none})"
  # --stop-file は共通の $STOP: ユーザーが触ると進行中のランも層境界で止まる
  if [ -n "$tmo" ]; then
    timeout "$tmo" "$PY" -u experiment.py "$@" --checkpoint "$ck" --stop-file "$STOP" > "$lg" 2>&1
  else
    "$PY" -u experiment.py "$@" --checkpoint "$ck" --stop-file "$STOP" > "$lg" 2>&1
  fi
  local rc=$? best
  best=$(grep "best hard test acc" "$lg" | tail -1)
  log "DONE $name (exit=$rc) : $best"
}

log "=== night batch start (stop: touch $STOP) ==="
COMMON="--dataset mnist --device cuda --batch 512 --group-residual --skip-input --max-layers 120 --patience 20 --skip-e2e"

run carryskip_s2 ""  $COMMON --window 4 --commit 1 --carry --seed 2
run carryskip_s3 ""  $COMMON --window 4 --commit 1 --carry --seed 3
run convfunnel   4h  --dataset mnist --device cuda --batch 2048 --group-residual --conv-sched 64,64,128 --conv-pool-sched 2,2,1 --conv-tree 3 --epochs 120 --patience 10 --max-layers 40 --skip-e2e --seed 1
run thresholds   3h  $COMMON --thresholds 31,63,127,191 --seed 1

log "=== night batch end ==="
