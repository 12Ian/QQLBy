param(
    [string]$PythonPath = "",
    [string]$OutputPath = ""
)
$ErrorActionPreference = "Stop"
$moduleRoot = Split-Path $PSScriptRoot -Parent
$xuanceRoot = Split-Path $moduleRoot -Parent
$repositoryRoot = Split-Path $xuanceRoot -Parent
if (-not $PythonPath) { $PythonPath = Join-Path $repositoryRoot ".venv/Scripts/python.exe" }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Python missing: $PythonPath. Read README installation steps." }
if (-not $OutputPath) {
    $runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPath = Join-Path $xuanceRoot "outputs/trajectory/smoke_$runStamp"
}
$runData = Join-Path $OutputPath "data.npz"
$runTraining = Join-Path $OutputPath "pipeline"
$runEntry = Join-Path $moduleRoot "run.py"
$runConfig = Join-Path $moduleRoot "configs/cpu_completion.json"
& $PythonPath $runEntry check --device cpu --config $runConfig --backward
if ($LASTEXITCODE -ne 0) { throw "Environment check failed" }
& $PythonPath $runEntry generate --output $runData --tracks 90 --steps 96
if ($LASTEXITCODE -ne 0) { throw "Data generation failed" }
& $PythonPath $runEntry pipeline --data $runData --config $runConfig --output $runTraining --intention-epochs 3
if ($LASTEXITCODE -ne 0) { throw "Training failed" }
$runCheckpoint = Join-Path $runTraining "forecast/best.pt"
& $PythonPath $runEntry predict --data $runData --checkpoint $runCheckpoint --track-id track_000000 --device cpu --output (Join-Path $runTraining "prediction.csv")
if ($LASTEXITCODE -ne 0) { throw "Prediction failed" }
& $PythonPath $runEntry rolling-evaluate --data $runData --checkpoint $runCheckpoint --device cpu --refresh-steps 1 --output (Join-Path $OutputPath "rolling")
if ($LASTEXITCODE -ne 0) { throw "Rolling evaluation failed" }
Write-Output "CPU smoke completed: $OutputPath"
