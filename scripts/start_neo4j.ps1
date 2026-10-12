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

if (Test-Neo4jPort) {
    Write-Host "Neo4j is already available at bolt://localhost:7687"
    exit 0
}

$workspace = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).Path)
$neo4j = Join-Path $workspace ".runtime\neo4j\neo4j-community-$Version\bin\neo4j.bat"
if (-not (Test-Path -LiteralPath $neo4j)) {
    throw "Neo4j is not installed. Run .\scripts\setup_neo4j.ps1 first."
}

$pidFile = Join-Path $workspace ".runtime\neo4j\console-process.pid"
$consoleProcess = Start-Process `
    -FilePath $env:ComSpec `
    -ArgumentList @("/d", "/c", "`"$neo4j`" console") `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $consoleProcess.Id -Encoding ascii
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-Neo4jPort) {
        Write-Host "Neo4j started: bolt://localhost:7687 (Browser: http://localhost:7474)"
        exit 0
    }
    Start-Sleep -Seconds 1
}
if (Test-Path -LiteralPath $pidFile) {
    Remove-Item -LiteralPath $pidFile -Force
}
throw "Neo4j did not open Bolt port 7687. Check .runtime\neo4j\neo4j-community-$Version\logs\neo4j.log"

