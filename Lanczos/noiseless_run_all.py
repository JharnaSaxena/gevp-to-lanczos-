#!/usr/bin/env python3
"""
noiseless_run_all.py single orchestrator
Runs the three project files in the correct order:
  1. lanczos_core.py      (library loads cleanly)
  2. eq24_comparison.py   (eq. 24 audit + side-by-side, writes output/*.txt)
  3. figures.py           (figs 2, 3, 6, 7 + fig2_data.csv, writes output/*.txt)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["lanczos_core.py", "eq24_comparison.py", "figures.py"]

def main() -> int:
    for name in SCRIPTS:
        path = HERE / name
        if not path.exists():
            print(f"[skip] {name} not found in {HERE}")
            continue

        print()
        print("=" * 72)
        print(f"  {name}")
        print("=" * 72)

        result = subprocess.run([sys.executable, str(path)], cwd=HERE)
        if result.returncode != 0:
            print(f"\n[FAIL] {name} exited with code {result.returncode}")
            return result.returncode

    print("\n"*10 + "+"*90)
    print("  all done — logs & figures under:", HERE / "output")
    return 0


if __name__ == "__main__":
    sys.exit(main())