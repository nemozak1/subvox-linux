"""Head-orientation calibration for `subvox spatial`.

The motion frame gives a gyroscope with an unknown scale, a gravity vector, and four
orientation words with unknown meaning. This module records a scripted set of head moves
(left/right 90°, up/down, tilt) and works out:

  * the gyro scale (rad/s per LSB) from the 90° turns and from the gravity-measured tilts,
  * the head axes in the sensor frame: `up` (gravity at rest), `left` (axis the head nods
    about), `forward` = left × up,
  * which orientation word, if any, tracks yaw / pitch / roll and at what scale.

Results go to ~/.local/share/subvox/head-calibration.json, which `subvox spatial` loads.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

CALIBRATION_PATH = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "subvox/head-calibration.json"
RECORDINGS_DIR = CALIBRATION_PATH.parent / "calibration"

# (label, instruction, seconds). "hold" phases are analysed; centre phases reset.
PHASES = [
    ("still", "Sit still, look straight ahead at where the music should come from", 4),
    ("left", "Turn your head LEFT 90° (chin over left shoulder) and hold", 4),
    ("centre", "Back to centre", 3),
    ("right", "Turn your head RIGHT 90° and hold", 4),
    ("centre", "Back to centre", 3),
    ("up", "Look UP (nose to the ceiling) and hold", 4),
    ("centre", "Back to centre", 3),
    ("down", "Look DOWN (chin to chest) and hold", 4),
    ("centre", "Back to centre", 3),
    ("tilt_left", "Tilt your head LEFT (left ear toward left shoulder) and hold", 4),
    ("centre", "Back to centre", 3),
    ("tilt_right", "Tilt your head RIGHT and hold", 4),
    ("centre", "Back to centre and sit still", 4),
]
HOLDS = ("left", "right", "up", "down", "tilt_left", "tilt_right")
CSV_FIELDS = ["t", "phase", "counter", "o0", "o1", "o2", "o3", "gx", "gy", "gz", "ax", "ay", "az"]

Vec = tuple[float, float, float]


def _norm(v) -> Vec:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _mean(rows, keys) -> list[float]:
    return [sum(float(r[k]) for r in rows) / len(rows) for k in keys]


def _angle_deg(a, b) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(_norm(a), _norm(b))))))


@dataclass
class HeadCalibration:
    """What `subvox spatial` needs. Defaults are the uncalibrated guesses."""

    gyro_rad_per_lsb: float = 1e-3
    gyro_bias: Vec = (0.0, 0.0, 0.0)
    up: Vec | None = None          # sensor-frame unit vectors with the head centred
    left: Vec | None = None
    forward: Vec | None = None
    orient: dict = field(default_factory=dict)  # {"yaw": {"index": 2, "deg_per_lsb": x}, ...}
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "HeadCalibration":
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        c = cls(gyro_rad_per_lsb=d.get("gyro_rad_per_lsb", 1e-3), gyro_bias=tuple(d.get("gyro_bias", (0, 0, 0))),
                orient=d.get("orient", {}), notes=d.get("notes", []))
        for k in ("up", "left", "forward"):
            if d.get(k):
                setattr(c, k, tuple(d[k]))
        return c

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"gyro_rad_per_lsb": self.gyro_rad_per_lsb, "gyro_bias": list(self.gyro_bias),
             "up": list(self.up) if self.up else None, "left": list(self.left) if self.left else None,
             "forward": list(self.forward) if self.forward else None, "orient": self.orient, "notes": self.notes}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# ---------------------------------------------------------------- recording

def write_recording(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_recording(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- analysis

def _segments(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Consecutive runs of the same phase, in order."""
    segs: list[tuple[str, list[dict]]] = []
    for r in rows:
        if segs and segs[-1][0] == r["phase"]:
            segs[-1][1].append(r)
        else:
            segs.append((r["phase"], [r]))
    return segs


def _integrate_gyro(rows: list[dict], bias: Vec) -> Vec:
    """∫(gyro - bias) dt over the rows, in LSB·s, per axis."""
    acc = [0.0, 0.0, 0.0]
    for a, b in zip(rows, rows[1:]):
        dt = float(b["t"]) - float(a["t"])
        if not 0 < dt < 0.5:
            continue
        for i, k in enumerate(("gx", "gy", "gz")):
            acc[i] += (float(a[k]) - bias[i]) * dt
    return tuple(acc)


def analyze(rows: list[dict]) -> tuple[HeadCalibration, str]:
    """Return the calibration and a human-readable report."""
    segs = _segments(rows)
    out: list[str] = []
    cal = HeadCalibration()

    still = next((s for p, s in segs if p == "still"), None)
    if not still or len(still) < 10:
        raise ValueError("recording has no 'still' phase")
    bias = tuple(_mean(still, ("gx", "gy", "gz")))
    up = _norm(_mean(still, ("ax", "ay", "az")))
    o_base = _mean(still, ("o0", "o1", "o2", "o3"))
    cal.gyro_bias = bias
    cal.up = up
    out.append(f"gyro bias (LSB): {tuple(round(b, 1) for b in bias)}   up (sensor frame): {tuple(round(u, 3) for u in up)}")
    out.append(f"orientation words at rest: {[round(o) for o in o_base]}")

    # Per hold: settled mean (last 60 % of the hold), gravity tilt, integrated gyro over the move into the hold.
    holds: dict[str, dict] = {}
    for i, (phase, seg) in enumerate(segs):
        if phase not in HOLDS:
            continue
        settled = seg[len(seg) * 2 // 5:]
        prev = segs[i - 1][1] if i > 0 else []
        move_rows = prev[len(prev) // 2:] + seg          # from mid-centre into the hold
        rot = _integrate_gyro(move_rows, bias)
        g = _mean(settled, ("ax", "ay", "az"))
        holds[phase] = {
            "orient": [a - b for a, b in zip(_mean(settled, ("o0", "o1", "o2", "o3")), o_base)],
            "tilt_deg": _angle_deg(g, up),
            "rot": rot,
            "rot_mag": math.sqrt(_dot(rot, rot)),
            "axis": _norm(rot),
        }
    out.append("")
    out.append(f"{'hold':11} {'tilt°':>6} {'∫gyro LSB·s':>12}  axis (sensor frame)       Δwords")
    for p in HOLDS:
        if p in holds:
            h = holds[p]
            out.append(f"{p:11} {h['tilt_deg']:6.1f} {h['rot_mag']:12.0f}  "
                       f"({h['axis'][0]:+.2f} {h['axis'][1]:+.2f} {h['axis'][2]:+.2f})   "
                       f"{[round(o) for o in h['orient']]}")

    # Gyro scale: 90° turns (nominal) and gravity-measured tilts (actual).
    scales_90 = [math.radians(90) / holds[p]["rot_mag"] for p in ("left", "right") if p in holds and holds[p]["rot_mag"] > 0]
    scales_g = [math.radians(holds[p]["tilt_deg"]) / holds[p]["rot_mag"]
                for p in ("up", "down", "tilt_left", "tilt_right")
                if p in holds and holds[p]["rot_mag"] > 0 and holds[p]["tilt_deg"] > 10]
    out.append("")
    if scales_g:
        cal.gyro_rad_per_lsb = sum(scales_g) / len(scales_g)
        out.append(f"gyro scale from gravity-measured tilts: {cal.gyro_rad_per_lsb:.3e} rad/s per LSB "
                   f"({', '.join(f'{s:.3e}' for s in scales_g)})")
        if scales_90:
            out.append(f"  cross-check from the nominal 90° turns: {', '.join(f'{s:.3e}' for s in scales_90)} "
                       f"(turn measured as {', '.join(f'{90 * cal.gyro_rad_per_lsb / s:.0f}°' for s in scales_90)})")
    elif scales_90:
        cal.gyro_rad_per_lsb = sum(scales_90) / len(scales_90)
        out.append(f"gyro scale from the 90° turns: {cal.gyro_rad_per_lsb:.3e} rad/s per LSB")
    else:
        out.append("could not estimate the gyro scale; keeping the default")

    # Axes. Yaw axis should be gravity; nod axis is the head's left-right axis.
    if "left" in holds and "right" in holds:
        yaw_axis = _norm([a - b for a, b in zip(holds["left"]["rot"], holds["right"]["rot"])])  # left minus right
        d = _dot(yaw_axis, up)
        out.append(f"yaw axis · up = {d:+.3f} (±1 is ideal; + means a left turn is a positive rotation about up)")
        if d < 0:
            cal.notes.append("left turn is a negative rotation about measured up: left-handed gyro frame, yaw sign flipped")
    if "up" in holds and "down" in holds:
        nod = [a - b for a, b in zip(holds["down"]["rot"], holds["up"]["rot"])]   # down minus up
        # x forward, y left, z up: a positive rotation about +left tips the nose down.
        left = _norm(nod)
        left = _norm([l - _dot(left, up) * u for l, u in zip(left, up)])  # orthogonalise against up
        cal.left = left
        cal.forward = _norm(_cross(left, up))
        out.append(f"left axis (sensor frame): {tuple(round(x, 3) for x in left)}   "
                   f"forward: {tuple(round(x, 3) for x in cal.forward)}   left·up = {_dot(left, up):+.3f}")
        if "tilt_left" in holds and "tilt_right" in holds:
            roll_axis = _norm([a - b for a, b in zip(holds["tilt_right"]["rot"], holds["tilt_left"]["rot"])])
            out.append(f"roll axis · forward = {_dot(roll_axis, cal.forward):+.3f} (+1 is ideal)")

    # Orientation words: which one moves with which rotation, and by how much per degree.
    out.append("")
    for name, pos, neg, angle in (("yaw", "left", "right", None), ("pitch", "up", "down", None),
                                  ("roll", "tilt_left", "tilt_right", None)):
        if pos not in holds or neg not in holds:
            continue
        deg = 90.0 if name == "yaw" else (holds[pos]["tilt_deg"] + holds[neg]["tilt_deg"]) / 2
        if deg < 5:
            continue
        best, best_score = None, 0.0
        for idx in range(4):
            dp, dn = holds[pos]["orient"][idx], holds[neg]["orient"][idx]
            # Want opposite-signed, similar-sized deltas; score = antisymmetric part over symmetric part.
            anti, sym = (dp - dn) / 2, abs(dp + dn) / 2
            score = abs(anti) / (sym + 50)
            others = max(abs(holds[q]["orient"][idx]) for q in holds if q not in (pos, neg)) if len(holds) > 2 else 0
            out.append(f"  {name:5} word{idx}: Δ{pos}={dp:+.0f} Δ{neg}={dn:+.0f} | other holds max {others:.0f} | score {score:.1f}")
            if score > best_score and abs(anti) > 200:
                best, best_score = (idx, anti / deg), score
        if best and best_score > 2:
            idx, deg_per = best
            cal.orient[name] = {"index": idx, "deg_per_lsb": deg / (abs(deg_per) * deg) if deg_per else 0,
                                "sign": 1 if deg_per > 0 else -1}
            cal.orient[name]["deg_per_lsb"] = 1 / abs(deg_per)
            out.append(f"  -> {name} = word{idx} × {cal.orient[name]['deg_per_lsb']:.4f}°/LSB "
                       f"(sign {cal.orient[name]['sign']:+d}; 180/32768 = {180 / 32768:.4f}, 360/65536 same)")
        else:
            out.append(f"  -> no orientation word clearly tracks {name}")
    return cal, "\n".join(out)
