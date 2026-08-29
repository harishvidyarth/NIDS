# Windows: install deps (first run only) then start the NIDS backend + UI.
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo "backend\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $repo "backend\.venv")
}

& $py -m pip install -r (Join-Path $repo "backend\requirements.txt")
& $py (Join-Path $repo "scripts\patch_cicflowmeter.py") $py

Set-Location $repo
& $py -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8765
