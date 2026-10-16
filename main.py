"""One-click entry point to run the full pipeline.

Runs in order: preprocess -> woe -> scorecard -> challenger -> validate.
Each script runs as a separate subprocess so it keeps its own __main__
context (needed to unpickle the fitted WoEBinner).

Usage:
    python main.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # points to the venv python when the venv is active

STEPS = ["preprocess.py", "woe.py", "scorecard.py", "challenger.py", "validate.py"]


def main() -> None:
    start = time.time()
    for step in STEPS:
        print(f"\n{'=' * 30} {step} {'=' * 30}")
        subprocess.run([PYTHON, str(ROOT / "src" / step)], check=True)
    print(f"\nPipeline finished in {time.time() - start:.1f}s. See output/.")


if __name__ == "__main__":
    main()
