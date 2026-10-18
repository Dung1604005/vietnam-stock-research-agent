param(
    [string]$TaskName = "VietnamStockResearchSync",
    [string]$At = "09:00",
    [switch]$Remove
)
$ErrorActionPreference = "Stop"
$researchWorkspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([IO.Path]::GetPathRoot($researchWorkspace) -ne "D:\") {
    throw "Research scheduler must run from drive D."
}
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    return
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw "Task already exists. Inspect it or remove this exact task before reinstalling."
}
$researchPython = Join-Path $researchWorkspace ".venv\Scripts\python.exe"
$researchWindowlessPython = Join-Path $researchWorkspace ".venv\Scripts\pythonw.exe"
if (Test-Path -LiteralPath $researchWindowlessPython) {
    $researchPython = $researchWindowlessPython
}
$researchScript = Join-Path $researchWorkspace "scripts\sync_research_corpus.py"
if (!(Test-Path -LiteralPath $researchPython) -or !(Test-Path -LiteralPath $researchScript)) {
    throw "Workspace Python or sync script is missing."
}
$researchAction = New-ScheduledTaskAction -Execute $researchPython -Argument ('"' + $researchScript + '" --due') -WorkingDirectory $researchWorkspace
$researchTrigger = New-ScheduledTaskTrigger -Daily -At $At
$researchSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
$researchPrincipal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $researchAction -Trigger $researchTrigger -Settings $researchSettings -Principal $researchPrincipal -Description "Discover public issuer reports and index approved documents into local Research RAG."
