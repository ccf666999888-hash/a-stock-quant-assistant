from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
subprocess.run([sys.executable, str(root / "scripts" / "generate_daily_report.py"), *sys.argv[1:]], cwd=root, check=True)

