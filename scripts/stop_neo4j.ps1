param([string]$Version = "2026.09.0")

$ErrorActionPreference = "Stop"
function Test-Neo4jPort {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", 7687)
        return $task.Wait(500) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

$workspace = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).Path)
$neo4j = Join-Path $workspace ".runtime\neo4j\neo4j-community-$Version\bin\neo4j.bat"
$pidFile = Join-Path $workspace ".runtime\neo4j\console-process.pid"
if (Test-Path -LiteralPath $pidFile) {
    $consolePid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
    $consoleProcess = Get-Process -Id $consolePid -ErrorAction SilentlyContinue
    if ($consoleProcess -and $consoleProcess.ProcessName -ne "cmd") {
        throw "Refusing to stop PID $consolePid because it is not the Neo4j console wrapper."
    }
    if ($consoleProcess) {
        & taskkill.exe /PID $consolePid /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Windows denied stopping Neo4j PID $consolePid. Re-run this script from the same terminal privilege level used to start it."
        }
    }
    Remove-Item -LiteralPath $pidFile -Force
}
elseif (Test-Path -LiteralPath $neo4j) {
    & $neo4j stop
}

for ($attempt = 0; $attempt -lt 15; $attempt++) {
    if (-not (Test-Neo4jPort)) {
        Write-Host "Neo4j stopped. Project data remains in the workspace on D:."
        exit 0
    }
    Start-Sleep -Seconds 1
}

throw "Neo4j is still listening on port 7687."

