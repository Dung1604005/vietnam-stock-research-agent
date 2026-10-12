$ErrorActionPreference = "Stop"
$workspace = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).Path)
$pidFile = Join-Path $workspace ".runtime\ollama-project-server.pid"

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "No project Ollama PID file was found."
    exit 0
}

$serverPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$serverProcess = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($serverProcess -and $serverProcess.ProcessName -ne "cmd") {
    throw "Refusing to stop PID $serverPid because it is not the project Ollama wrapper."
}
if ($serverProcess) {
    & taskkill.exe /PID $serverPid /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Windows denied stopping project Ollama PID $serverPid."
    }
}
Remove-Item -LiteralPath $pidFile -Force
Write-Host "Project Ollama stopped. Models remain on drive D."
