# タスク28: conv の MNIST 審判(一晩バッチ)。
# 相手は Vol.1 手法看板 MNIST 94.64%(単発500・残差+skip・3シード平均)。
#
# 設計:
#  - 6GB VRAM なので直列。1本ずつ順に流し、各本に --checkpoint を持たせる
#    (層コミット単位で保存 = 途中で切れても翌晩に続きから再開できる)。
#  - 各本に個別の時間キャップ。超過したら --stop-file を置いて「進行中の層を
#    終えてから」きれいに停止させる(プロセスをkillしない = 層が無駄にならない)。
#  - バッチ全体の予算を超えたら残りはスキップ(翌晩、同じスクリプトで再開)。
#  - epochs は checkpoint の fingerprint 対象なので、120 と 240 は別の .pt。
$ErrorActionPreference = "Continue"
$root = "C:\work\github_Mming-Lab\greedy-lgn"
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$batchLog = "runs\task28_batch.log"

$GLOBAL_BUDGET_H = 8.5      # バッチ全体の上限(時間)
$GRACE_MIN = 40             # stop-file を置いてから諦めてkillするまでの猶予(分)

# 共通条件: 残差readout・patience 10(タスク29の教訓=深さを十分探す)・
# batch 2048・seed 1。conv は simplify 未対応なので e2e ともどもスキップ
$common = @("--dataset", "mnist", "--device", "cuda", "--batch", "2048",
            "--group-residual", "--patience", "10", "--max-layers", "40",
            "--skip-e2e", "--seed", "1")

# 実行順: 安い順。4本目(C32@240)は時間が余ったときだけ = 「convはMNISTでも
# 240エポック要るか」の検証(digits掃引の外挿を実地確認する軸)
$runs = @(
  @{ name = "task28_c32";       cap = 2.5; args = @("--conv", "32", "--conv-tree", "3", "--epochs", "120") },
  @{ name = "task28_c64";       cap = 3.0; args = @("--conv", "64", "--conv-tree", "3", "--epochs", "120") },
  @{ name = "task28_funnel";    cap = 3.0; args = @("--conv-sched", "128,64,32", "--conv-tree", "3", "--epochs", "120") },
  @{ name = "task28_c32_ep240"; cap = 3.0; args = @("--conv", "32", "--conv-tree", "3", "--epochs", "240") }
)

function Say($msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  Write-Host $line
  Add-Content -Path $batchLog -Value $line
}

$batchStart = Get-Date
Say "=== task28 conv batch start (global budget ${GLOBAL_BUDGET_H}h) ==="

foreach ($r in $runs) {
  $elapsed = ((Get-Date) - $batchStart).TotalHours
  $left = $GLOBAL_BUDGET_H - $elapsed
  if ($left -le 0.2) { Say "SKIP $($r.name): global budget exhausted (elapsed $([Math]::Round($elapsed,2))h)"; continue }

  $log = "runs\$($r.name).log"
  $ckpt = "runs\$($r.name).pt"
  $stop = "runs\$($r.name).stop"
  if (Test-Path $stop) { Remove-Item $stop }   # 前回の停止合図が残っていたら消す

  $cap = [Math]::Min($r.cap, $left)
  # -u = 非バッファ出力。リダイレクト先でも層ごとの進捗がすぐログに出る
  # (バッファのままだと途中経過が見えず、killした本はログごと消える)
  $argv = @("-u", "experiment.py") + $common + $r.args + @("--checkpoint", $ckpt, "--stop-file", $stop)
  Say "START $($r.name) (cap $([Math]::Round($cap,2))h) : $($argv -join ' ')"

  $p = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -PassThru `
                     -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  $deadline = (Get-Date).AddHours($cap)
  $stopped = $false
  while (-not $p.HasExited) {
    Start-Sleep -Seconds 30
    if (-not $stopped -and (Get-Date) -gt $deadline) {
      # 時間切れ: 進行中の層を終えてから止まってもらう(次の層の頭で効く)
      New-Item -ItemType File -Path $stop | Out-Null
      $stopped = $true
      $kill = (Get-Date).AddMinutes($GRACE_MIN)
      Say "CAP $($r.name): stop-file placed, waiting for a clean layer boundary"
    }
    if ($stopped -and (Get-Date) -gt $kill) {
      Say "KILL $($r.name): grace expired (checkpoint keeps the committed layers)"
      Stop-Process -Id $p.Id -Force
      break
    }
  }
  $mins = [Math]::Round(((Get-Date) - $batchStart).TotalMinutes, 1)
  $best = (Select-String -Path $log -Pattern "best hard test acc" | Select-Object -Last 1).Line
  Say "DONE $($r.name) (exit=$($p.ExitCode)) : $best"
}

Say "=== task28 conv batch end (total $([Math]::Round(((Get-Date) - $batchStart).TotalHours,2))h) ==="
