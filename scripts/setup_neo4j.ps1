param(
    [string]$Version = "2026.09.0",
    [string]$Password = "stock-agent-local-neo4j"
)

$ErrorActionPreference = "Stop"
$workspace = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).Path)
$downloadDirectory = Join-Path $workspace ".cache\downloads"
$archive = Join-Path $downloadDirectory "neo4j-community-$Version-windows.zip"
$installRoot = Join-Path $workspace ".runtime\neo4j"
$installDirectory = Join-Path $installRoot "neo4j-community-$Version"

New-Item -ItemType Directory -Force -Path $downloadDirectory, $installRoot | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri "https://dist.neo4j.org/neo4j-community-$Version-windows.zip" -OutFile $archive -UseBasicParsing
}
if (-not (Test-Path -LiteralPath $installDirectory)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $installRoot -Force
}

$configPath = Join-Path $installDirectory "conf\neo4j.conf"
$config = Get-Content -LiteralPath $configPath -Raw
$config = $config -replace "(?m)^#?dbms\.usage_report\.enabled=.*$", "dbms.usage_report.enabled=false"
$config = $config -replace "(?m)^#?server\.memory\.heap\.initial_size=.*$", "server.memory.heap.initial_size=256m"
$config = $config -replace "(?m)^#?server\.memory\.heap\.max_size=.*$", "server.memory.heap.max_size=512m"
$config = $config -replace "(?m)^#?server\.memory\.pagecache\.size=.*$", "server.memory.pagecache.size=256m"
Set-Content -LiteralPath $configPath -Value $config -Encoding utf8

$systemDatabase = Join-Path $installDirectory "data\databases\system"
if (-not (Test-Path -LiteralPath $systemDatabase)) {
    & "$installDirectory\bin\neo4j-admin.bat" dbms set-initial-password $Password
}

Write-Host "Neo4j installed on drive D at: $installDirectory"
Write-Host "Run: .\scripts\start_neo4j.ps1"

