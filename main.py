"""一键跑通全流程的入口。

按顺序执行：preprocess -> woe -> scorecard -> challenger -> validate。
每个脚本作为独立进程运行，保持各自的运行环境一致。

用法：
    python main.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # 当前解释器（激活 venv 后即指向 venv 的 python）

STEPS = ["preprocess.py", "woe.py", "scorecard.py", "challenger.py", "validate.py"]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 30} {step} {'=' * 30}")
        subprocess.run([PYTHON, str(ROOT / "src" / step)], check=True)
    print("\n全流程完成。")


if __name__ == "__main__":
    main()
