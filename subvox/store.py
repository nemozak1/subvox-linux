"""Training set on disk, like the Mac app's `dataset.json`."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .classifier import Template

DEFAULT_PATH = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "subvox/dataset.json"
SESSIONS_DIR = Path.home() / "Documents/Subvox/Sessions"


def load(path: Path = DEFAULT_PATH) -> list[Template]:
    if not path.exists():
        return []
    with open(path) as f:
        return [Template.from_dict(d) for d in json.load(f)]


def save(templates: list[Template], path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump([t.to_dict() for t in templates], f)
    tmp.replace(path)
