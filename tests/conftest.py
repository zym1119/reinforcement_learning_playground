import shutil
from pathlib import Path

import pytest

WORK_DIRS = Path(__file__).resolve().parent.parent / 'work_dirs'


def pytest_sessionfinish(session, exitstatus):
    """测试结束后自动删除 work_dirs 中以 _test_ 开头的实验目录"""
    if WORK_DIRS.exists():
        for d in WORK_DIRS.iterdir():
            if d.is_dir() and d.name.startswith('_test_'):
                shutil.rmtree(d)
