param(
    [string]$HostAddress = "127.0.0.1:11435"
)

$ErrorActionPreference = "Stop"
$workspace = [System.IO.Path]::GetFullPath((Resolve-Path (Join-Path $PSScriptRoot "..")).Path)
$modelDirectory = Join-Path $workspace ".runtime\ollama-models"
$ollamaExecutable = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$pidFile = Join-Path $workspace ".runtime\ollama-project-server.pid"
$baseUrl = "http://$HostAddress"

if (-not (Test-Path -LiteralPath $ollamaExecutable)) {
    throw "Ollama executable not found at $ollamaExecutable"
}
if (-not (Test-Path -LiteralPath $modelDirectory)) {
    throw "Ollama model directory not found at $modelDirectory"
}

try {
    $tags = Invoke-RestMethod -Uri "$baseUrl/api/tags" -TimeoutSec 2
    if ($tags.models.Count -gt 0) {
        Write-Host "Project Ollama is already available at $baseUrl"
        $tags.models | ForEach-Object { Write-Host "- $($_.name)" }
        exit 0
    }
    throw "Port $HostAddress is occupied by an Ollama server with no project models."
}
catch [System.Net.WebException] {
    # Expected when the dedicated server is not running yet.
}

$command = "set `"OLLAMA_MODELS=$modelDirectory`" && set `"OLLAMA_HOST=$HostAddress`" && `"$ollamaExecutable`" serve"
$server = Start-Process `
    -FilePath $env:ComSpec `
    -ArgumentList @("/d", "/c", $command) `
    -WindowStyle Hidden `
    -PassThru
Set-Content -LiteralPath $pidFile -Value $server.Id -Encoding ascii

for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $tags = Invoke-RestMethod -Uri "$baseUrl/api/tags" -TimeoutSec 2
        if ($tags.models.Count -gt 0) {
            Write-Host "Project Ollama started at $baseUrl"
            $tags.models | ForEach-Object { Write-Host "- $($_.name)" }
            exit 0
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

throw "Project Ollama did not start at $baseUrl"
