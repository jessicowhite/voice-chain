# One-time setup: venv + dependencies + desktop shortcut. Run from the repo folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\python -m pip install -r requirements.txt

$ws = New-Object -ComObject WScript.Shell
$s = $ws.CreateShortcut("$env:USERPROFILE\Desktop\Voice Chain.lnk")
$s.TargetPath = "$root\.venv\Scripts\pythonw.exe"
$s.Arguments = "app.py"
$s.WorkingDirectory = $root
$s.Description = "Voice Chain - noise removal dashboard"
$s.Save()

Write-Host "Done. Launch 'Voice Chain' from the Desktop."
if (-not (Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue | Where-Object FriendlyName -match 'CABLE')) {
    Write-Host "NOTE: VB-Cable virtual audio device not found. Install it from https://vb-audio.com/Cable/ (needed as the output)."
}
