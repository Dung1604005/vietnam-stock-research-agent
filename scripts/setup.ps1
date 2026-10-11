$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ((Split-Path -Qualifier $projectRoot) -ne "D:") {
    throw "Workspace rule: install this project from drive D:. Current path: $projectRoot"
}

$projectTemp = Join-Path $projectRoot ".tmp"
$pipCache = Join-Path $projectRoot ".cache\pip"
$virtualEnv = Join-Path $projectRoot ".venv"
New-Item -ItemType Directory -Force -Path $projectTemp, $pipCache | Out-Null
$env:TEMP = $projectTemp
$env:TMP = $projectTemp
$env:PYTHONUTF8 = "1"
$env:PIP_PROGRESS_BAR = "off"

if (-not (Test-Path (Join-Path $virtualEnv "Scripts\python.exe"))) {
    python -m venv $virtualEnv
}

& (Join-Path $virtualEnv "Scripts\python.exe") -m pip install `
    --progress-bar off `
    --cache-dir $pipCache `
    --extra-index-url "https://vnstocks.com/api/simple" `
    -e "${projectRoot}[dev]"

Write-Host "Installed into $virtualEnv"
