"""Runtime settings and status for `subvox spatial`, shared with the desktop controls.

Settings live in ~/.config/subvox/spatial.json and are reloaded by the daemon whenever the
file changes, so the Waybar module and the settings popup only ever edit the file. The daemon
writes its state to ~/.local/state/subvox/spatial-status.json twice a second.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

SETTINGS_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "subvox/spatial.json"
STATUS_PATH = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "subvox/spatial-status.json"

DEFAULTS: dict = {
    "enabled": False,     # desired state; `subvox ctl on/off` keeps the audio graph in step
    "linger": 6.0,        # s: time constant for the stage to drift back to where you face (0 = never)
    "lead": 0.2,          # s: aim at the head pose this far ahead (A2DP ~0.21 + quantum 0.02)
    "deadband": 1.5,      # °: only rebuild the HRIRs when a speaker moved this much
    "spread": 30.0,       # °: speaker half-angle
    "elevation": True,    # follow pitch/roll as well as yaw
    "rate": 12.5,         # max control updates per second
}
LIMITS: dict[str, tuple[float, float]] = {
    "linger": (0.0, 60.0), "lead": (0.0, 0.5), "deadband": (0.0, 10.0), "spread": (5.0, 90.0), "rate": (2.0, 25.0),
}


class Settings:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path
        self.data = dict(DEFAULTS)
        self._mtime: float | None = None
        self.reload()

    def reload(self) -> bool:
        """Re-read the file if it changed. Returns True when values were (re)loaded."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            with open(self.path) as f:
                loaded = json.load(f)
        except (OSError, ValueError):
            return False
        self.data = dict(DEFAULTS)
        for k, v in loaded.items():
            if k in DEFAULTS:
                self.data[k] = self.coerce(k, v)
        return True

    @staticmethod
    def coerce(key: str, value):
        if isinstance(DEFAULTS[key], bool):
            return bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "on", "yes")
        v = float(value)
        if key in LIMITS:
            lo, hi = LIMITS[key]
            v = max(lo, min(hi, v))
        return v

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        tmp.replace(self.path)
        self._mtime = self.path.stat().st_mtime

    def set(self, key: str, value) -> None:
        if key not in DEFAULTS:
            raise KeyError(key)
        self.data[key] = self.coerce(key, value)

    def __getitem__(self, key: str):
        return self.data[key]


def write_status(**fields) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields["t"] = time.time()
    tmp = STATUS_PATH.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(fields, f)
    tmp.replace(STATUS_PATH)


def read_status(max_age: float = 3.0) -> dict:
    """The daemon's last status, or {"stream": "stopped"} if it is stale or missing."""
    try:
        with open(STATUS_PATH) as f:
            d = json.load(f)
        if time.time() - d.get("t", 0) > max_age:
            return {"stream": "stopped"}
        return d
    except (OSError, ValueError):
        return {"stream": "stopped"}
