# Daily prediction (run after market close)
# Usage:
#   .\scripts\daily_predict.ps1
#   .\scripts\daily_predict.ps1 -Date 2026-05-19

param(
    [string]$Date = ""
)

Set-Location $PSScriptRoot\..

$MODEL  = "lgbm_optuna"
$FREQ   = "daily"
$PYTHON = "C:\ProgramData\miniconda3\envs\qlib_zhengshi\python.exe"

$pyArgs = @(
    "python/predict_live.py",
    "--model", $MODEL,
    "--freq",  $FREQ,
    "--mode",  "predict-only"
)
if ($Date -ne "") {
    $pyArgs += @("--date", $Date)
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm"
Write-Host ""
Write-Host "=== Daily predict ($MODEL / $FREQ) $ts ===" -ForegroundColor Cyan

& $PYTHON @pyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "predict_live failed (exit=$LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($Date -ne "") {
    $outDir = "predictions\$Date"
} else {
    $outDir = "predictions\<latest-trading-day>"
}
Write-Host ""
Write-Host "Output dir: $outDir" -ForegroundColor Green
