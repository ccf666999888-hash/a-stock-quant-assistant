$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ is required and must be available as 'python'."
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip -i $Mirror
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt -i $Mirror

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

& ".\.venv\Scripts\python.exe" scripts/setup_database.py
Write-Host "Initialization complete. Set TUSHARE_TOKEN in .env before syncing data."
