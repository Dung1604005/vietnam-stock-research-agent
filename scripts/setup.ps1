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
$env:PIP_CACHE_DIR = $pipCache
$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
$env:TORCH_HOME = Join-Path $projectRoot ".cache\torch"
$env:XDG_CACHE_HOME = Join-Path $projectRoot ".cache"
$env:STOCK_AGENT_RUNTIME_DIR = Join-Path $projectRoot ".runtime"

if (-not (Test-Path (Join-Path $virtualEnv "Scripts\python.exe"))) {
    python -m venv $virtualEnv
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the D-drive virtual environment." }
}

& (Join-Path $virtualEnv "Scripts\python.exe") -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11 or newer is required'"
if ($LASTEXITCODE -ne 0) { throw "Use Python 3.11 or newer to bootstrap .venv." }

& (Join-Path $virtualEnv "Scripts\python.exe") -m pip install `
    --progress-bar off `
    --cache-dir $pipCache `
    --extra-index-url "https://vnstocks.com/api/simple" `
    -e "${projectRoot}[dev]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. The app has not been verified." }

Write-Host "Installed into $virtualEnv"
Write-Host "Optional TensorFlow/E5/LLM training packages were not installed and no model was downloaded."
Write-Host "Start: .\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py"
Write-Host "Demo profiles, private replay capture and sanitized exports: docs/demo-runbook.md"
