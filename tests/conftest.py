"""
项目级 pytest conftest：把 python/ 加入 sys.path，便于
测试通过绝对导入 `from backtest.strategy import ...`。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
