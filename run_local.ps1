# Lance plateforme-decision-sante-v2 dans son venv ISOLÉ (.venv).
# Code isolé   : run_api.py / dashboard épinglent la racine de CE projet à sys.path.
# Build isolé  : utilise exclusivement .venv\ (jamais le Python global).
# Ports        : API 8010, dashboard 8501.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$st = Join-Path $root ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $py)) { throw "venv absent : exécute d'abord  python -m venv .venv ; .venv\Scripts\python -m pip install -r requirements.txt" }

Write-Host "[v2] API       -> http://127.0.0.1:8010/docs"
Write-Host "[v2] Dashboard -> http://127.0.0.1:8501"

# API en arrière-plan (fenêtre séparée), dashboard au premier plan.
Start-Process -FilePath $py -ArgumentList "run_api.py", "--port", "8010" -WorkingDirectory $root
& $st run (Join-Path $root "dashboard\app.py") --server.port 8501 --server.headless true
