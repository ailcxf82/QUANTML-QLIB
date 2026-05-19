# Monthly retrain (run after Qlib data update, ~30-60 min)
#
# Steps:
#   1. Freeze Qlib snapshot (robocopy)
#   2. Update configs/freq/daily_retrain.yaml provider_uri (manual)
#   3. Ensemble gate backtest (3-seed + daily_vol_target, no model.pkl)
#   4. Single-seed retrain (writes latest_*.pkl for predict_live)
#   5. Promote to live path (latest_*_daily.pkl)
#
# Usage:
#   .\scripts\monthly_retrain.ps1
#   .\scripts\monthly_retrain.ps1 -SkipSnapshot
#   .\scripts\monthly_retrain.ps1 -SkipSnapshot -SkipGate

param(
    [switch]$SkipSnapshot,
    [switch]$SkipGate
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
