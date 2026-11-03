"""Head tracker geometry against a synthetic bud: the regression for "tilt back -> one ear"."""

import math
import unittest

from subvox import spatial as sp
from subvox.calibrate import HeadCalibration
from subvox.motion import ACCEL_LSB_PER_G, MotionSample

# Nemo's calibration (2026-11-02): sensor-frame head axes with the bud as it sat that day.
CAL = HeadCalibration(up=(0.6706, -0.7319, -0.1209), left=(-0.3648, -0.1834, -0.9129),
                      forward=(-0.6459, -0.6563, 0.3899), world_up=(0.0, 0.0, 1.0))


def rot(axis, deg):
    a = sp.norm(axis)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return lambda v: tuple(v[i] * c + sp.cross(a, v)[i] * s + a[i] * sp.dot(a, v) * (1 - c) for i in range(3))


def mat(f):
    """3x3 (rows) of a linear map given as a function."""
    cols = [f((1, 0, 0)), f((0, 1, 0)), f((0, 0, 1))]
    return [[cols[j][i] for j in range(3)] for i in range(3)]


def quat_from_mat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1) * 2
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return q if q[0] >= 0 else tuple(-x for x in q)


class SyntheticBud:
    """A bud whose calibrated mounting is CAL, pivoted `pivot_deg` about its insertion (sensor z)
    axis. Frames come from a head attitude given as yaw/pitch/roll (deg, left/nose-up/right-ear-down)."""

    def __init__(self, pivot_deg: float):
        m0 = lambda v: tuple(CAL.forward[i] * v[0] + CAL.left[i] * v[1] + CAL.up[i] * v[2] for i in range(3))
        rz = rot((0, 0, 1), pivot_deg)
        self.head_to_sensor = lambda v: rz(m0(v))                     # M1
        self.ear_axis = self.head_to_sensor((0, 1, 0))
        m1 = mat(self.head_to_sensor)
        self.sensor_to_head = lambda v: tuple(sum(m1[j][i] * v[j] for j in range(3)) for i in range(3))
        self.t = 0.0

    def frame(self, yaw=0.0, pitch=0.0, roll=0.0) -> MotionSample:
        head = lambda v: rot((0, 0, 1), yaw)(rot((0, 1, 0), -pitch)(rot((1, 0, 0), roll)(v)))   # head -> world
        head_t = lambda v: tuple(sum(mat(head)[j][i] * v[j] for j in range(3)) for i in range(3))
        q = quat_from_mat(mat(lambda v: head(self.sensor_to_head(v))))                           # sensor -> world
        down = self.head_to_sensor(head_t((0.0, 0.0, -1.0)))
        acc = tuple(int(round(x * ACCEL_LSB_PER_G)) for x in down)
        words = tuple(int(round(x * 32768)) for x in q[1:])
        self.t += 0.04
        return MotionSample(self.t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, raw_accel=acc, raw_orient=(*words, 0))


def run(tracker, bud, poses):
    out = []
    for p in poses:
        pose = tracker.update(bud.frame(*p))
        if pose:
            out.append(pose)
    return out


def ramp(a, b, n):
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(sp.dot(sp.norm(a), sp.norm(b)))))))


class TrackerGeometry(unittest.TestCase):
    def test_rotate_min(self):
        v = sp.rotate_min((0, 0, 1), (1, 0, 0), (0, 0, 1))
        self.assertAlmostEqual(sp.dot(v, (1, 0, 0)), 1.0, places=6)
        v = sp.rotate_min((0, 0, 1), (1, 0, 0), (0, 1, 0))     # the rotation axis stays put
        self.assertAlmostEqual(sp.dot(v, (0, 1, 0)), 1.0, places=6)

    def test_pitch_back_with_pivoted_bud_keeps_the_stage_in_front(self):
        bud = SyntheticBud(pivot_deg=90)
        tr = sp.HeadTracker(CAL)
        run(tr, bud, [(0, 0, 0)] * 60)                         # level and still: W, seating, front
        self.assertTrue(tr.ref_set)
        self.assertEqual(tr.ear_axis_source, "seating")
        self.assertLess(angle(tr.L_dev, bud.ear_axis), 15)
        poses = run(tr, bud, [(0, p, 0) for p in ramp(0, 60, 25)] + [(0, 60, 0)] * 25)
        worst = max(abs(p.yaw) for p in poses)
        self.assertLess(worst, 15, f"yaw wandered to {worst:.1f}° while tilting back")
        self.assertAlmostEqual(poses[-1].pitch, 60, delta=3)
        az_l, _ = tr.speaker_direction(30)
        az_r, _ = tr.speaker_direction(-30)
        self.assertLess(abs(sp.wrap180(az_l - 30)), 15)
        self.assertLess(abs(sp.wrap180(az_r + 30)), 15)

    def test_nods_refine_the_ear_axis_without_moving_the_stage(self):
        bud = SyntheticBud(pivot_deg=90)
        tr = sp.HeadTracker(CAL)
        run(tr, bud, [(0, 0, 0)] * 60)
        before = angle(tr.L_dev, bud.ear_axis)
        nods = []
        for _ in range(3):
            nods += [(0, p, 0) for p in ramp(0, -35, 10)] + [(0, p, 0) for p in ramp(-35, 0, 10)]
        run(tr, bud, nods)
        poses = run(tr, bud, [(0, 0, 0)] * 60)                 # still again: refinement runs here
        self.assertEqual(tr.ear_axis_source, "nods")
        self.assertLess(angle(tr.L_dev, bud.ear_axis), 2, f"ear axis still {angle(tr.L_dev, bud.ear_axis):.1f}° off (was {before:.1f}°)")
        self.assertLess(max(abs(p.yaw) for p in poses), 2, "the stage jumped when the axis was adopted")
        poses = run(tr, bud, [(0, p, 0) for p in ramp(0, 60, 25)] + [(0, 60, 0)] * 10)
        self.assertLess(max(abs(p.yaw) for p in poses), 2)
        poses = run(tr, bud, [(y, 0, 0) for y in ramp(0, 40, 25)] + [(40, 0, 0)] * 10)
        self.assertAlmostEqual(poses[-1].yaw, 40, delta=2)
        poses = run(tr, bud, [(40, 0, r) for r in ramp(0, 25, 10)] + [(40, 0, 25)] * 5)
        self.assertAlmostEqual(poses[-1].yaw, 40, delta=2)      # rolling the head does not turn the stage
        self.assertAlmostEqual(poses[-1].roll, 25, delta=3)

    def test_unpivoted_bud_is_exact(self):
        bud = SyntheticBud(pivot_deg=0)
        tr = sp.HeadTracker(CAL)
        run(tr, bud, [(0, 0, 0)] * 60)
        poses = run(tr, bud, [(0, p, 0) for p in ramp(0, 60, 25)] + [(-30, 60, 0)] * 10)
        self.assertLess(max(abs(p.yaw) for p in poses[:25]), 1)
        self.assertAlmostEqual(poses[-1].yaw, -30, delta=1)


if __name__ == "__main__":
    unittest.main()
