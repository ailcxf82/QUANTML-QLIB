# Monthly retrain (run after Qlib data update, ~30-60 min)
#
# Steps:
#   1.  Freeze Qlib snapshot (robocopy)
#   1b. Update configs/freq/daily_retrain.yaml provider_uri (manual)
#   1c. Auto-advance dataset_segments + base.yaml end_time to latest data
#       (was hardcoded to 2020-2023, see docs/lessons; without 1c retrain just
#        refits the same 2020-2023 window and predict_live keeps loading a
#        2023-12-31 model with newer saved_at timestamp)
#   2.  Backup current production pkl
#   3.  Ensemble gate backtest (3-seed + daily_vol_target, no model.pkl)
#   4.  Single-seed retrain (writes latest_*.pkl for predict_live)
#   5.  Promote to live path (latest_*_daily.pkl)
#
# Usage:
#   .\scripts\monthly_retrain.ps1
#   .\scripts\monthly_retrain.ps1 -SkipSnapshot
#   .\scripts\monthly_retrain.ps1 -SkipSnapshot -SkipGate
#   .\scripts\monthly_retrain.ps1 -DataEnd 2026-05-25
#   .\scripts\monthly_retrain.ps1 -SkipUpdateSegments   # legacy behavior

param(
    [switch]$SkipSnapshot,
    [switch]$SkipGate,
    [switch]$SkipUpdateSegments,
    [string]$DataEnd = ""
)

Set-Location $PSScriptRoot\..

$MODEL   = "lgbm_optuna"
$FREQ    = "daily_retrain"
$PYTHON  = "C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe"
$MONTH   = Get-Date -Format "yyyyMMdd"
$SRC     = "D:\qlib_data\qlib_data"
$DEST    = "D:\qlib_data\qlib_data_train_$MONTH"
$ARCHIVE = "artifacts\models\_archive"

# Step 1: freeze snapshot
if (-not $SkipSnapshot) {
    Write-Host ""
    Write-Host "[Step 1] Freeze snapshot -> $DEST" -ForegroundColor Cyan
    robocopy $SRC $DEST /MIR /MT:16 /NFL /NDL /NJH /NJS /R:1 /W:1
    if ($LASTEXITCODE -ge 8) {
        Write-Host "robocopy failed (exit=$LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  snapshot done" -ForegroundColor Green
    Write-Host ""
    Write-Host "[Step 1b] Update configs/freq/daily_retrain.yaml provider_uri to:"
    Write-Host "  D:/qlib_data/qlib_data_train_$MONTH" -ForegroundColor Yellow
    Read-Host "  Press Enter after editing yaml"
} else {
    Write-Host ""
    Write-Host "[Step 1] skipped (-SkipSnapshot)" -ForegroundColor DarkGray
}

# Step 1c: advance dataset_segments + base.yaml end_time
#
# Why this exists:
#   Both configs/freq/daily_retrain.yaml and configs/base.yaml hardcode train_end
#   to 2023-12-31. Without this step, "monthly retrain" effectively reruns the
#   exact same 2020-2023 fit and the resulting latest_lgbm_optuna_daily.pkl
#   keeps a stale train_end while saved_at moves forward, masking concept drift.
#   See docs/lessons/2026-05-11_p1p2_neutralize_pitfall.md for the rule
#   "training end must roll forward each retrain".
if (-not $SkipUpdateSegments) {
    Write-Host ""
    Write-Host "[Step 1c] Advance dataset_segments + base.yaml end_time" -ForegroundColor Cyan
    $updateArgs = @("scripts/update_retrain_segments.py", "--from-retrain-provider")
    if ($DataEnd -ne "") {
        # 显式指定时覆盖自动读取（去掉 --from-retrain-provider）
        $updateArgs = @("scripts/update_retrain_segments.py", "--data-end", $DataEnd)
    }
    # PYTHONIOENCODING=utf-8 ensures Chinese summary lines render correctly in PowerShell
    $env:PYTHONIOENCODING = "utf-8"
    & $PYTHON @updateArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "update_retrain_segments failed (exit=$LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  segments advanced; backups in configs/_archive/" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[Step 1c] skipped (-SkipUpdateSegments)  WARNING: train_end will stay fixed" -ForegroundColor Yellow
}

# Step 2: backup current model
Write-Host ""
Write-Host "[Step 2] Backup model -> $ARCHIVE" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $ARCHIVE -Force | Out-Null
$pklSrc  = "artifacts\models\latest_${MODEL}_${FREQ}.pkl"
$metaSrc = "artifacts\models\latest_${MODEL}_${FREQ}_meta.json"
if (Test-Path $pklSrc) {
    Copy-Item $pklSrc "$ARCHIVE\${MODEL}_${MONTH}_pre_retrain.pkl"
}
if (Test-Path $metaSrc) {
    Copy-Item $metaSrc "$ARCHIVE\${MODEL}_${MONTH}_pre_retrain_meta.json"
}
Write-Host "  backup done" -ForegroundColor Green

# Step 3: ensemble gate backtest
if (-not $SkipGate) {
    Write-Host ""
    Write-Host "[Step 3] Ensemble gate (3-seed + daily_vol_target)..." -ForegroundColor Cyan
    Write-Host "  ETA: 30-50 minutes"
    & $PYTHON python/run_experiment.py `
        --model $MODEL `
        --freq $FREQ `
        --ensemble-seeds 3 `
        --aggregator topk_vote `
        --strategy-cfg configs/strategy/daily_vol_target.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Gate backtest failed" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "Review artifacts/${MODEL}_${FREQ}_*/metrics.json" -ForegroundColor Yellow
    Write-Host '  sharpe:       prev - 0.3 or better'
    Write-Host '  ic_mean:      0.025 or higher'
    Write-Host '  icir:         0.13 or higher'
    Write-Host '  max_drawdown: -0.30 or higher (less negative)'
    $ok = Read-Host "  Pass gate? (y/N)"
    if ($ok -notmatch '^[Yy]') {
        Write-Host "Gate rejected, abort deploy" -ForegroundColor Red
        exit 0
    }
} else {
    Write-Host ""
    Write-Host "[Step 3] skipped (-SkipGate)" -ForegroundColor DarkGray
}

# Step 4: single-seed retrain (persist model)
Write-Host ""
Write-Host "[Step 4] Single-seed retrain -> latest_*.pkl" -ForegroundColor Cyan
Write-Host "  ETA: 5-10 minutes"
& $PYTHON python/run_experiment.py --model $MODEL --freq $FREQ
if ($LASTEXITCODE -ne 0) {
    Write-Host "Single-seed retrain failed" -ForegroundColor Red
    exit 1
}

# Step 5: promote to live path
Write-Host ""
Write-Host "[Step 5] Promote to live path" -ForegroundColor Cyan
$retrainPkl  = "artifacts\models\latest_${MODEL}_${FREQ}.pkl"
$retrainMeta = "artifacts\models\latest_${MODEL}_${FREQ}_meta.json"
$livePkl     = "artifacts\models\latest_${MODEL}_daily.pkl"
$liveMeta    = "artifacts\models\latest_${MODEL}_daily_meta.json"

if (-not (Test-Path $retrainPkl)) {
    Write-Host "Missing: $retrainPkl (check Step 4)" -ForegroundColor Red
    exit 1
}

Copy-Item -Force $retrainPkl $livePkl
if (Test-Path $retrainMeta) {
    Copy-Item -Force $retrainMeta $liveMeta
}
Write-Host "  updated: $livePkl" -ForegroundColor Green

Write-Host ""
Write-Host "=== Monthly retrain done ===" -ForegroundColor Green
Write-Host "  Run .\scripts\daily_predict.ps1 after market close"
