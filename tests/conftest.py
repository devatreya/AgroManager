from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import project_root


def load_task(split: str = "validation", index: int = 0) -> dict:
    path = project_root() / "data" / "processed" / f"scenario_tasks_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))[index]


@pytest.fixture
def sample_task() -> dict:
    return load_task()
