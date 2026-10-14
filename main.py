"""一键跑通全流程的入口。

按顺序执行：preprocess -> woe -> scorecard -> challenger -> validate。
用子进程逐个运行，保证每个脚本都以 __main__ 身份运行（反序列化 WoEBinner 的 pickle 需要）。

用法：
    python main.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # 激活 venv 后即指向 venv 的 python

STEPS = ["preprocess.py", "woe.py", "scorecard.py", "challenger.py", "validate.py"]


def main() -> None:
    start = time.time()
    for step in STEPS:
        print(f"\n{'=' * 30} {step} {'=' * 30}")
        subprocess.run([PYTHON, str(ROOT / "src" / step)], check=True)
    print(f"\n全流程完成，耗时 {time.time() - start:.1f} 秒。结果见 output/。")


if __name__ == "__main__":
    main()
