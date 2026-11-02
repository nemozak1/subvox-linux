"""Head-orientation calibration for `subvox spatial`.

The motion frame carries a gravity vector, a coarse gyroscope and four "orientation" words.
Found 2026-11-02 (Powerbeats Pro 2, fw 0167): words 0-2 / 32768 are the vector part of a unit
quaternion q (w = sqrt(1 - |v|²)) that maps sensor-frame vectors into a gravity-aligned world
frame: q·accel is the same world vector in every head pose (5° scatter vs 12° raw). Word 3 stays
near 0. That is Apple's fused attitude, yaw included, so `subvox spatial` uses it directly.
The accelerometer points along gravity (down), so head "up" = -accel; with that, a left turn is
a positive rotation about up, a right tilt positive about forward, and q·accel = (0, 0, -1),
i.e. world Z is up. All right-handed.

This module records a scripted set of head moves (left/right 90°, up/down, tilt) and works out:

  * the head axes in the sensor frame at the centred pose: `up` (gravity), `left` (the axis
    the head nods about, from the quaternion change between centre and look-up/down),
    `forward` = left × up,
  * the quaternion's world up vector and the sign of a left turn about it,
  * the gyro scale (rad/s per LSB) by comparing the raw gyro integral with the quaternion
    rotation over each move (fallback for `spatial --gyro`).

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
    gyro_sign: int = 1             # -1 if the gyro frame is left-handed w.r.t. the accelerometer
    up: Vec | None = None          # sensor-frame unit vectors with the head centred
    left: Vec | None = None
    forward: Vec | None = None
    world_up: Vec | None = None    # where q maps the accelerometer at rest (≈ (0, 0, -1))
    orient: dict = field(default_factory=dict)  # {"quaternion": True} when words 0-2 are a quaternion
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "HeadCalibration":
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        c = cls(gyro_rad_per_lsb=d.get("gyro_rad_per_lsb", 1e-3), gyro_bias=tuple(d.get("gyro_bias", (0, 0, 0))),
                gyro_sign=int(d.get("gyro_sign", 1)), orient=d.get("orient", {}), notes=d.get("notes", []))
        for k in ("up", "left", "forward", "world_up"):
            if d.get(k):
                setattr(c, k, tuple(d[k]))
        return c

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {"gyro_rad_per_lsb": self.gyro_rad_per_lsb, "gyro_bias": list(self.gyro_bias), "gyro_sign": self.gyro_sign,
             "up": list(self.up) if self.up else None, "left": list(self.left) if self.left else None,
             "forward": list(self.forward) if self.forward else None,
             "world_up": list(self.world_up) if self.world_up else None, "orient": self.orient, "notes": self.notes}
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


# ---------------------------------------------------------------- quaternion helpers

Quat = tuple[float, float, float, float]


def quat_from_words(o0: int, o1: int, o2: int) -> Quat | None:
    """Unit quaternion (w, x, y, z) from orientation words 0-2, or None if |v| > 1."""
    v = (o0 / 32768.0, o1 / 32768.0, o2 / 32768.0)
    n = _dot(v, v)
    if n > 1.0001:
        return None
    return (math.sqrt(max(0.0, 1.0 - n)), *v)


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def qrot(q: Quat, v) -> Vec:
    p = qmul(qmul(q, (0.0, v[0], v[1], v[2])), qconj(q))
    return (p[1], p[2], p[3])


def qmean(qs: list[Quat]) -> Quat:
    """Component mean with sign made consistent, renormalised. Fine for a cluster of poses."""
    ref = qs[0]
    acc = [0.0] * 4
    for q in qs:
        if sum(a * b for a, b in zip(q, ref)) < 0:
            q = tuple(-x for x in q)
        acc = [a + b for a, b in zip(acc, q)]
    n = math.sqrt(sum(x * x for x in acc)) or 1.0
    return tuple(x / n for x in acc)


def qaxis_angle(q: Quat) -> tuple[Vec, float]:
    """Rotation axis (unit) and angle in degrees."""
    w = max(-1.0, min(1.0, q[0]))
    v = q[1:]
    s = math.sqrt(max(0.0, 1 - w * w))
    if s < 1e-9:
        return (0.0, 0.0, 1.0), 0.0
    ang = 2 * math.degrees(math.acos(abs(w)))
    axis = tuple(x / s for x in v)
    if w < 0:
        axis = tuple(-x for x in axis)
    return axis, ang


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


def _row_quat(r: dict) -> Quat | None:
    return quat_from_words(int(r["o0"]), int(r["o1"]), int(r["o2"]))


def _seg_quat(rows: list[dict]) -> Quat:
    qs = [q for q in (_row_quat(r) for r in rows) if q is not None]
    return qmean(qs)


def analyze(rows: list[dict]) -> tuple[HeadCalibration, str]:
    """Return the calibration and a human-readable report."""
    segs = _segments(rows)
    out: list[str] = []
    cal = HeadCalibration()

    still = next((s for p, s in segs if p == "still"), None)
    if not still or len(still) < 10:
        raise ValueError("recording has no 'still' phase")
    still = still[-100:]   # the last 4 s: anything earlier predates the first prompt
    bias = tuple(_mean(still, ("gx", "gy", "gz")))
    up = _norm([-x for x in _mean(still, ("ax", "ay", "az"))])   # accel points down
    cal.gyro_bias = bias
    cal.up = up

    # Quaternion sanity: q·accel must be one fixed world vector in every pose.
    world_ups = []
    for r in rows:
        q = _row_quat(r)
        if q is None:
            continue
        a = (int(r["ax"]) / 1024, int(r["ay"]) / 1024, int(r["az"]) / 1024)
        if 0.85 < math.sqrt(_dot(a, a)) < 1.15:
            world_ups.append(qrot(q, a))
    wu = _norm([-sum(u[i] for u in world_ups) / len(world_ups) for i in range(3)])
    scatter = sum(_angle_deg(u, tuple(-x for x in wu)) for u in world_ups) / len(world_ups)
    raw_mean = _mean(rows, ("ax", "ay", "az"))
    raw_scatter = sum(_angle_deg((int(r["ax"]), int(r["ay"]), int(r["az"])), raw_mean) for r in rows) / len(rows)
    cal.world_up = wu
    cal.orient = {"quaternion": True, "world_up_scatter_deg": round(scatter, 1)}
    out.append(f"gyro bias (LSB): {tuple(round(b, 1) for b in bias)}   up (sensor frame, centred): {tuple(round(u, 3) for u in up)}")
    out.append(f"quaternion check: world up = -q·accel = ({wu[0]:+.3f} {wu[1]:+.3f} {wu[2]:+.3f}) with {scatter:.1f}° mean scatter "
               f"(raw accel scatters {raw_scatter:.1f}°) -> words 0-2 {'ARE' if scatter < raw_scatter / 1.5 else 'are NOT'} a quaternion")

    # Per hold: rotation from the preceding centre pose, in the world frame and the centre's sensor frame.
    holds: dict[str, dict] = {}
    for i, (phase, seg) in enumerate(segs):
        if phase not in HOLDS or i == 0:
            continue
        prev = segs[i - 1][1]
        q_c = _seg_quat(prev[len(prev) * 2 // 5:])
        q_h = _seg_quat(seg[len(seg) * 2 // 5:])
        r_world = qmul(q_h, qconj(q_c))             # rotation that took centre to hold, world frame
        axis_w, ang = qaxis_angle(r_world)
        axis_dev = qrot(qconj(q_c), axis_w)          # same axis in the sensor frame at centre
        move_rows = prev[len(prev) // 2:] + seg
        gyro = _integrate_gyro(move_rows, bias)
        g = [-x for x in _mean(seg[len(seg) * 2 // 5:], ("ax", "ay", "az"))]
        holds[phase] = {"angle": ang, "axis_w": axis_w, "axis_dev": axis_dev, "q_c": q_c,
                        "tilt_deg": _angle_deg(g, up), "gyro_mag": math.sqrt(_dot(gyro, gyro))}
    out.append("")
    out.append(f"{'hold':11} {'angle°':>7} {'grav tilt°':>10}  axis (sensor frame @ centre)   axis·up   ∫gyro LSB·s")
    for p in HOLDS:
        if p in holds:
            h = holds[p]
            ad = h["axis_dev"]
            out.append(f"{p:11} {h['angle']:7.1f} {h['tilt_deg']:10.1f}  ({ad[0]:+.2f} {ad[1]:+.2f} {ad[2]:+.2f})          "
                       f"{_dot(ad, up):+.2f}   {h['gyro_mag']:8.0f}")

    # Gyro scale: quaternion rotation angle vs raw integral, per move.
    scales = [math.radians(h["angle"]) / h["gyro_mag"] for h in holds.values() if h["gyro_mag"] > 5 and h["angle"] > 15]
    out.append("")
    if scales:
        scales.sort()
        cal.gyro_rad_per_lsb = scales[len(scales) // 2]
        out.append(f"gyro scale (median of {len(scales)}): {cal.gyro_rad_per_lsb:.3e} rad/s per LSB  "
                   f"[{', '.join(f'{x:.2e}' for x in scales)}]")
    else:
        out.append("could not estimate the gyro scale; keeping the default")

    # Yaw: a left turn should be a rotation about world up. Record the sign.
    if "left" in holds and "right" in holds:
        d_l = _dot(holds["left"]["axis_w"], wu)
        d_r = _dot(holds["right"]["axis_w"], wu)
        out.append(f"left turn axis · world_up = {d_l:+.2f}, right turn {d_r:+.2f} (opposite signs, |·| near 1 expected)")
        out.append(f"turn sizes: left {holds['left']['angle']:.0f}°, right {holds['right']['angle']:.0f}° (asked for 90°)")
        if d_l < 0:
            cal.notes.append("left turn is a negative rotation about world up: unexpected handedness")

    # Head axes from the nod: left = -axis(look up) = +axis(look down) (x fwd, y left, z up).
    if "up" in holds and "down" in holds:
        nod = _norm([-a + b for a, b in zip(holds["up"]["axis_dev"], holds["down"]["axis_dev"])])
        left = _norm([l - _dot(nod, up) * u for l, u in zip(nod, up)])
        cal.left = left
        cal.forward = _norm(_cross(left, up))
        out.append(f"left axis (sensor frame): {tuple(round(x, 3) for x in left)}   forward: {tuple(round(x, 3) for x in cal.forward)}"
                   f"   nod axis · up = {_dot(nod, up):+.2f} (0 ideal)")
        out.append(f"up/down sizes: {holds['up']['angle']:.0f}° / {holds['down']['angle']:.0f}°")
        if "tilt_left" in holds and "tilt_right" in holds:
            roll = _norm([a - b for a, b in zip(holds["tilt_right"]["axis_dev"], holds["tilt_left"]["axis_dev"])])
            out.append(f"roll axis · forward = {_dot(roll, cal.forward):+.2f} (+1 ideal; tilt sizes "
                       f"{holds['tilt_left']['angle']:.0f}° / {holds['tilt_right']['angle']:.0f}°)")
        if "left" in holds:
            # sign check: yaw axis in sensor frame should be ±up
            out.append(f"yaw axis · up (sensor frame) = {_dot(holds['left']['axis_dev'], up):+.2f}")
    return cal, "\n".join(out)
