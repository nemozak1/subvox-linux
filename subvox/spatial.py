"""Head-tracked spatial audio: keep the virtual speakers of the "Spatial (Powerbeats)" sink
fixed in the room while the head turns.

Pipeline: HeadphoneMotion (25 Hz) -> HeadTracker (the buds' own fused attitude quaternion, see
calibrate.py; `--gyro` integrates the coarse gyro instead) -> 10 Hz: where are the ±30° speakers
seen from the head? -> pw-cli set-param on the filter-chain sink's spatializer controls
(Azimuth / Elevation per speaker).

Like Apple's head tracking the reference frame drifts back to wherever the head has been
pointing for a while, so looking away briefly moves the stage and settling in recentres it.

Latency: the A2DP sink buffers ~214 ms and the graph quantum adds ~21 ms, so a control change is
heard ~240 ms late. The renderer therefore aims the speakers at where the head will be `lead`
seconds from now, extrapolated from the smoothed body angular velocity. Overshoot when the head
stops is bounded by |ω|·lead and decays within one lead time.
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from .calibrate import HeadCalibration, quat_from_words
from .motion import ACCEL_LSB_PER_G, HeadphoneMotion, MotionSample

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w x y z

SINK_NODE = "subvox_spatial"
SPEAKERS = {"spFL": +1, "spFR": -1}   # sign of the azimuth spread


# ---------------------------------------------------------------- small vector / quaternion kit

def norm(v) -> Vec:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qconj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q: Quat) -> Quat:
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return tuple(x / n for x in q)


def qrot(q: Quat, v: Vec) -> Vec:
    """Rotate v by q."""
    p = qmul(qmul(q, (0.0, *v)), qconj(q))
    return (p[1], p[2], p[3])


def qfrom_rotvec(r: Vec) -> Quat:
    """Quaternion for a rotation vector (axis * angle)."""
    th = math.sqrt(dot(r, r))
    if th < 1e-12:
        return (1.0, r[0] / 2, r[1] / 2, r[2] / 2)
    s = math.sin(th / 2) / th
    return (math.cos(th / 2), r[0] * s, r[1] * s, r[2] * s)


def qslerp(a: Quat, b: Quat, t: float) -> Quat:
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b, d = tuple(-x for x in b), -d
    if d > 0.9995:
        return qnorm(tuple(x + t * (y - x) for x, y in zip(a, b)))
    th = math.acos(d)
    sa, sb = math.sin((1 - t) * th) / math.sin(th), math.sin(t * th) / math.sin(th)
    return tuple(sa * x + sb * y for x, y in zip(a, b))


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- head tracker

@dataclass
class Pose:
    yaw: float      # degrees, left positive
    pitch: float    # degrees, nose-up positive
    roll: float     # degrees, right-ear-down positive
    still: bool
    rate: float     # angular speed, deg/s


class HeadTracker:
    """Attitude from gyro + gravity, relative to a slowly recentring reference.

    `q` maps sensor-frame vectors to the room frame (room frame = sensor frame of the
    calibrated centred pose). `q_ref` is the reference the stage is anchored to.
    """

    def __init__(self, cal: HeadCalibration, gravity_tau: float = 2.0, recentre_tau: float = 6.0,
                 still_rate_deg: float = 4.0, still_after: float = 1.0, use_quaternion: bool = True) -> None:
        self.cal = cal
        self.use_quaternion = use_quaternion
        self.gravity_tau = gravity_tau
        self.recentre_tau = recentre_tau
        self.still_rate = math.radians(still_rate_deg)
        self.still_after = still_after
        self.q: Quat = (1.0, 0.0, 0.0, 0.0)
        self.q_ref: Quat = (1.0, 0.0, 0.0, 0.0)
        self.up: Vec | None = tuple(cal.up) if cal.up else None
        self.left: Vec | None = tuple(cal.left) if cal.left else None
        self.forward: Vec | None = tuple(cal.forward) if cal.forward else None
        self._t_prev: float | None = None
        self._still_since: float | None = None
        self._boot_acc: list[Vec] = []
        self.ready = self.up is not None
        self.rate = 0.0
        self.omega: Vec = (0.0, 0.0, 0.0)   # body angular velocity, rad/s, EMA (tau omega_tau)
        self.omega_tau = 0.06
        # Online gyro scale: angle the gravity vector turned through vs. the raw gyro integral
        # over the same window. Robust median of many nods/tilts. Off when the file is calibrated.
        self.auto_scale = cal.up is None
        self.scale = cal.gyro_rad_per_lsb
        self._win: list[tuple[float, Vec, Vec]] = []   # (t, up_dev, raw gyro - bias)
        self.scale_estimates: list[float] = []

    # -- frame
    def _init_frame(self, up: Vec) -> None:
        self.up = norm(up)
        if self.left is None:
            seed = (1.0, 0.0, 0.0) if abs(self.up[0]) < 0.9 else (0.0, 1.0, 0.0)
            left = [s - dot(seed, self.up) * u for s, u in zip(seed, self.up)]
            self.left = norm(left)
        self.forward = norm(cross(self.left, self.up))
        self.ready = True

    def recentre(self) -> None:
        self.q_ref = self.q

    # -- update
    def update(self, s: MotionSample) -> Pose | None:
        # The accelerometer points along gravity, so head "up" is -accel.
        a = tuple(-v / ACCEL_LSB_PER_G for v in s.raw_accel)
        a_mag = math.sqrt(dot(a, a))
        if not self.ready:
            # No calibration: the first second at rest defines "up".
            self._boot_acc.append(a)
            if len(self._boot_acc) >= 25:
                self._init_frame(tuple(sum(v[i] for v in self._boot_acc) / len(self._boot_acc) for i in range(3)))
            return None

        if self.use_quaternion:
            q = quat_from_words(*s.raw_orient[:3])
            if q is None:
                return None
            if self._t_prev is None:
                self.q = self.q_ref = q
                self._t_prev = s.t
                return None
            dt = s.t - self._t_prev
            self._t_prev = s.t
            if not 0 < dt < 0.5:
                return None
            dq = qmul(qconj(self.q), q)          # body-frame rotation since the last frame
            if dq[0] < 0:
                dq = tuple(-x for x in dq)
            ang = 2 * math.acos(min(1.0, dq[0]))
            s_half = math.sqrt(max(0.0, 1 - dq[0] * dq[0]))
            w_body = tuple(x / s_half * ang / dt for x in dq[1:]) if s_half > 1e-9 else (0.0, 0.0, 0.0)
            self.rate = ang / dt
            self._smooth_omega(w_body, dt)
            self.q = q
        else:
            if self._t_prev is None:
                self._t_prev = s.t
                return None
            dt = s.t - self._t_prev
            self._t_prev = s.t
            if not 0 < dt < 0.5:
                return None
            c = self.cal
            raw = tuple(g - b for g, b in zip(s.raw_gyro, c.gyro_bias))
            if self.auto_scale and 0.9 < a_mag < 1.1:
                self._learn_scale(s.t, norm(a), raw, dt)
            w = tuple(c.gyro_sign * r * self.scale for r in raw)
            self.rate = math.sqrt(dot(w, w))
            self._smooth_omega(w, dt)
            # integrate body rates
            self.q = qnorm(qmul(self.q, qfrom_rotvec(tuple(x * dt for x in w))))
            # gravity pulls pitch/roll back when the accelerometer reads ~1 g
            if 0.8 < a_mag < 1.2:
                u_room = qrot(self.q, norm(a))
                err = cross(u_room, self.up)
                k = min(1.0, dt / self.gravity_tau)
                self.q = qnorm(qmul(qfrom_rotvec(tuple(e * k for e in err)), self.q))

        # stillness + recentring
        now = s.t
        if self.rate < self.still_rate:
            if self._still_since is None:
                self._still_since = now
        else:
            self._still_since = None
        still = self._still_since is not None and now - self._still_since >= self.still_after
        if still and self.recentre_tau > 0:
            self.q_ref = qslerp(self.q_ref, self.q, min(1.0, dt / self.recentre_tau))

        return self.pose(still)

    def _learn_scale(self, t: float, u: Vec, raw: Vec, dt: float) -> None:
        self._win.append((t, u, tuple(r * dt for r in raw)))
        self._win = [e for e in self._win if e[0] >= t - 0.8]
        if len(self._win) < 5:
            return
        u0 = self._win[0][1]
        theta = math.acos(max(-1.0, min(1.0, dot(u0, u))))
        if theta < math.radians(8):
            return
        integ = tuple(sum(e[2][i] for e in self._win[1:]) for i in range(3))
        mid = norm(tuple(a + b for a, b in zip(u0, u)))
        perp = tuple(x - dot(integ, mid) * m for x, m in zip(integ, mid))   # only rotation ⟂ up tilts it
        mag = math.sqrt(dot(perp, perp))
        if mag <= 0:
            return
        est = theta / mag
        if 1e-5 < est < 1e-1:
            self.scale_estimates.append(est)
            self.scale_estimates = self.scale_estimates[-25:]
            if len(self.scale_estimates) >= 3:
                self.scale = sorted(self.scale_estimates)[len(self.scale_estimates) // 2]
        self._win.clear()

    def _smooth_omega(self, w: Vec, dt: float) -> None:
        k = 1.0 - math.exp(-dt / self.omega_tau)
        self.omega = tuple(o + k * (x - o) for o, x in zip(self.omega, w))

    # -- readouts
    def q_rel(self, lead: float = 0.0) -> Quat:
        """Head pose relative to the reference, extrapolated `lead` seconds ahead."""
        q = self.q
        if lead > 0:
            rv = tuple(x * lead for x in self.omega)
            mag = math.sqrt(dot(rv, rv))
            cap = math.radians(45)
            if mag > cap:
                rv = tuple(x * cap / mag for x in rv)
            q = qnorm(qmul(q, qfrom_rotvec(rv)))
        return qmul(qconj(self.q_ref), q)

    def pose(self, still: bool = False) -> Pose:
        qr = self.q_rel()
        fw = qrot(qr, self.forward)
        lf = qrot(qr, self.left)
        yaw = math.degrees(math.atan2(dot(fw, self.left), dot(fw, self.forward)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, dot(fw, self.up)))))
        roll = math.degrees(math.asin(max(-1.0, min(1.0, dot(lf, self.up)))))
        return Pose(yaw, pitch, roll, still, math.degrees(self.rate))

    def speaker_direction(self, azimuth_deg: float, elevation_deg: float = 0.0, lead: float = 0.0) -> tuple[float, float]:
        """Where a room-fixed source at (az, el) is, seen from the head `lead` s from now:
        (azimuth 0-360, elevation)."""
        az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
        s_ref = tuple(math.cos(el) * (math.cos(az) * f + math.sin(az) * l) + math.sin(el) * u
                      for f, l, u in zip(self.forward, self.left, self.up))
        s_dev = qrot(qconj(self.q_rel(lead)), s_ref)
        az_d = math.degrees(math.atan2(dot(s_dev, self.left), dot(s_dev, self.forward))) % 360.0
        el_d = math.degrees(math.asin(max(-1.0, min(1.0, dot(s_dev, self.up)))))
        return az_d, el_d


# ---------------------------------------------------------------- PipeWire control

def find_node(name: str = SINK_NODE) -> int | None:
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
        for o in json.loads(out):
            if o.get("info", {}).get("props", {}).get("node.name") == name:
                return o["id"]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


class SpatializerControl:
    """Drives the filter-chain controls through a long-lived `pw-cli` on a pipe."""

    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.proc: subprocess.Popen | None = None
        self.sent = 0

    def _ensure(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(["pw-cli"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, text=True)
        time.sleep(0.6)  # pw-cli must see the registry before it accepts ids

    def set(self, values: dict[str, float]) -> None:
        self._ensure()
        params = " ".join(f'"{k}" {v:.2f}' for k, v in values.items())
        try:
            self.proc.stdin.write(f"set-param {self.node_id} Props {{ params = [ {params} ] }}\n")
            self.proc.stdin.flush()
            self.sent += 1
        except (BrokenPipeError, OSError):
            self.proc = None

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                time.sleep(0.4)  # let the last set-param reach the server before quitting
                self.proc.stdin.write("quit\n"); self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


class StageRenderer:
    """Smooths and rate-limits the per-speaker azimuth/elevation and pushes changes."""

    def __init__(self, control: SpatializerControl, spread: float = 30.0, smooth_tau: float = 0.04,
                 min_change: float = 1.0, elevation: bool = True, lead: float = 0.2,
                 lead_min_rate_deg: float = 8.0) -> None:
        """Every change rebuilds both HRIRs and crossfades them over one quantum, which is audible
        as a faint flutter, so changes are only sent when a speaker moves by `min_change` degrees,
        and the lead prediction (which amplifies sensor jitter) is faded in above `lead_min_rate_deg`."""
        self.control = control
        self.spread = spread
        self.lead = lead
        self.lead_min_rate = lead_min_rate_deg
        self.smooth_tau = smooth_tau
        self.min_change = min_change
        self.elevation = elevation
        self.state: dict[str, tuple[float, float]] = {k: (sign * spread % 360.0, 0.0) for k, sign in SPEAKERS.items()}
        self.last_sent = dict(self.state)
        self._t_prev: float | None = None
        self.updates = 0

    def update(self, tracker: HeadTracker, now: float) -> bool:
        dt = 0.1 if self._t_prev is None else max(1e-3, now - self._t_prev)
        self._t_prev = now
        alpha = 1.0 - math.exp(-dt / self.smooth_tau) if self.smooth_tau > 0 else 1.0
        changed = False
        values: dict[str, float] = {}
        rate_deg = math.degrees(tracker.rate)
        lead = self.lead * max(0.0, min(1.0, (rate_deg - self.lead_min_rate) / self.lead_min_rate))
        for name, sign in SPEAKERS.items():
            az, el = tracker.speaker_direction(sign * self.spread, lead=lead)
            if not self.elevation:
                el = 0.0
            paz, pel = self.state[name]
            az = (paz + alpha * wrap180(az - paz)) % 360.0
            el = pel + alpha * (el - pel)
            self.state[name] = (az, el)
            saz, sel = self.last_sent[name]
            if abs(wrap180(az - saz)) >= self.min_change or abs(el - sel) >= self.min_change:
                changed = True
            values[f"{name}:Azimuth"] = az
            values[f"{name}:Elevation"] = max(-90.0, min(90.0, el))
        if changed:
            self.control.set(values)
            self.last_sent = dict(self.state)
            self.updates += 1
        return changed

    def reset(self) -> None:
        values = {}
        for name, sign in SPEAKERS.items():
            values[f"{name}:Azimuth"] = sign * self.spread % 360.0
            values[f"{name}:Elevation"] = 0.0
        self.control.set(values)


# ---------------------------------------------------------------- demo source (no buds needed)

class DemoTracker(HeadTracker):
    """Sweeps yaw ±60° at 0.15 Hz so the renderer / CPU can be tested without the buds."""

    def __init__(self) -> None:
        super().__init__(HeadCalibration())
        self._init_frame((0.0, 0.0, 1.0))
        self.t0 = time.monotonic()

    def tick(self) -> Pose:
        yaw = math.radians(60 * math.sin(2 * math.pi * 0.15 * (time.monotonic() - self.t0)))
        self.q = qfrom_rotvec(tuple(u * yaw for u in self.up))
        return self.pose(False)


def filter_chain_cpu() -> float | None:
    """CPU seconds consumed by the filter-chain pipewire process, for the demo's load readout."""
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "filter-chain"],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        with open(f"/proc/{int(out)}/stat") as f:
            fields = f.read().split(")")[-1].split()
        return (int(fields[11]) + int(fields[12])) / 100.0  # utime + stime, clock ticks -> s
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
