"""Port of SubvoxTests/DecodingTests.swift."""

import math
import random
import unittest

from subvox import dtw, features
from subvox.classifier import Template, TemplateClassifier, evaluate
from subvox.features import FeatureSet
from subvox.motion import MotionSample


def sample(t: float, rotation=(0.0, 0.0, 0.0)) -> MotionSample:
    return MotionSample(t=t, ax=0, ay=0, az=0, rx=rotation[0], ry=rotation[1], rz=rotation[2],
                        pitch=0, roll=0, yaw=0, gx=0, gy=0, gz=-1)


def take(frequency: float, axis: int, frames: int = 30, noise: float = 0.01, rng=random) -> list[MotionSample]:
    """A synthetic "word": a burst of rotation at `frequency` cycles over the segment."""
    out = []
    for i in range(frames):
        t = i / 25
        envelope = math.sin(math.pi * i / (frames - 1))
        v = 0.2 * envelope * math.sin(2 * math.pi * frequency * i / frames)
        rot = [rng.uniform(-noise, noise) for _ in range(3)]
        rot[axis] += v
        out.append(sample(t, tuple(rot)))
    return out


class DecodingTests(unittest.TestCase):
    def test_dtw_identical_is_zero_and_warp_tolerant(self):
        a = features.frames(take(2, 0, noise=0), FeatureSet.ROTATION)
        stretched = features.frames(take(2, 0, frames=40, noise=0), FeatureSet.ROTATION)
        other = features.frames(take(2, 1, noise=0), FeatureSet.ROTATION)
        self.assertAlmostEqual(dtw.distance(a, a), 0, places=12)
        self.assertLess(dtw.distance(a, stretched), dtw.distance(a, other))

    def test_classifier_separates_distinct_words(self):
        templates = []
        for _ in range(6):
            templates.append(Template("yes", take(2, 0)))
            templates.append(Template("no", take(3, 1)))
        clf = TemplateClassifier(templates, FeatureSet.all())
        p = clf.predict(take(3, 1), max_distance=10, min_confidence=0)
        self.assertEqual(p.best.label, "no")
        self.assertTrue(p.accepted)

    def test_evaluation_beats_chance_only_when_signal_exists(self):
        real, noise = [], []
        for _ in range(8):
            real.append(Template("yes", take(2, 0)))
            real.append(Template("no", take(3, 2)))
            noise.append(Template("yes", take(0, 0, noise=0.05)))
            noise.append(Template("no", take(0, 0, noise=0.05)))
        signal = evaluate(real, FeatureSet.all(), permutations=200, seed=1)
        self.assertGreater(signal.accuracy, 0.9)
        self.assertLess(signal.p_value, 0.01)
        null = evaluate(noise, FeatureSet.all(), permutations=200, seed=1)
        self.assertGreater(null.p_value, 0.01)

    def test_trim_drops_quiet_padding(self):
        quiet = [sample(i / 25) for i in range(20)]
        burst = take(2, 0, noise=0)
        for s in burst:
            s.t += 0.8
        tail = [sample(i / 25 + 2) for i in range(20)]
        trimmed = features.trim(quiet + burst + tail)
        self.assertLess(len(trimmed), 40)
        self.assertGreaterEqual(len(trimmed), 20)


if __name__ == "__main__":
    unittest.main()
