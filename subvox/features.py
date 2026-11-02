"""Port of Subvox/Decoding/MotionFeatures.swift."""

from __future__ import annotations

import math
from enum import Flag, auto

from .motion import MotionSample


class FeatureSet(Flag):
    """Which motion channels feed the classifier. Toggling these is the ablation experiment."""

    ACCELERATION = auto()
    ROTATION = auto()
    ATTITUDE = auto()

    @classmethod
    def all(cls) -> "FeatureSet":
        return cls.ACCELERATION | cls.ROTATION | cls.ATTITUDE

    @classmethod
    def parse(cls, text: str) -> "FeatureSet":
        """'acc,rot,att' or 'all'."""
        if text == "all":
            return cls.all()
        names = {"acc": cls.ACCELERATION, "rot": cls.ROTATION, "att": cls.ATTITUDE}
        fs = cls(0)
        for part in text.split(","):
            fs |= names[part.strip()]
        return fs

    def label(self) -> str:
        return "+".join(n for n, f in (("acc", FeatureSet.ACCELERATION), ("rot", FeatureSet.ROTATION),
                                        ("att", FeatureSet.ATTITUDE)) if f in self) or "none"


def energy(s: MotionSample) -> float:
    """Activity score for segmentation. The 5x weight puts jaw-motion accel and rotation on one scale."""
    rotation = math.sqrt(s.rx * s.rx + s.ry * s.ry + s.rz * s.rz)
    accel = math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az)
    return rotation + 5 * accel


def trim(samples: list[MotionSample], padding: int = 2) -> list[MotionSample]:
    """Drop quiet lead-in/lead-out so templates align on the articulation itself."""
    if len(samples) <= 6:
        return samples
    energies = [energy(s) for s in samples]
    ordered = sorted(energies)
    floor = ordered[len(ordered) // 5]
    peak = ordered[-1]
    if not peak > floor * 1.5:
        return samples
    threshold = floor + 0.15 * (peak - floor)
    above = [i for i, e in enumerate(energies) if e > threshold]
    if not above:
        return samples
    lo = max(0, above[0] - padding)
    hi = min(len(samples) - 1, above[-1] + padding)
    if hi - lo + 1 < 5:
        return samples
    return samples[lo:hi + 1]


def _wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def frames(samples: list[MotionSample], fs: FeatureSet) -> list[list[float]]:
    """Per-frame feature vectors, mean-centred per channel (removes posture/heading bias)."""
    if not samples:
        return []
    origin = samples[0]
    out: list[list[float]] = []
    for s in samples:
        v: list[float] = []
        if FeatureSet.ACCELERATION in fs:
            v += [s.ax, s.ay, s.az]
        if FeatureSet.ROTATION in fs:
            v += [s.rx, s.ry, s.rz]
        if FeatureSet.ATTITUDE in fs:
            v += [_wrapped(s.pitch - origin.pitch), _wrapped(s.roll - origin.roll), _wrapped(s.yaw - origin.yaw)]
        out.append(v)
    dims = len(out[0])
    for d in range(dims):
        mean = sum(f[d] for f in out) / len(out)
        for f in out:
            f[d] -= mean
    return out


def channel_scales(sequences: list[list[list[float]]]) -> list[float]:
    """Per-channel standard deviation across a corpus, so no channel dominates the distance."""
    first = next((seq[0] for seq in sequences if seq), None)
    if first is None:
        return []
    dims = len(first)
    total = [0.0] * dims
    total_sq = [0.0] * dims
    n = 0
    for seq in sequences:
        for f in seq:
            for d in range(dims):
                total[d] += f[d]
                total_sq[d] += f[d] * f[d]
            n += 1
    if n <= 1:
        return [1.0] * dims
    scales = []
    for d in range(dims):
        mean = total[d] / n
        scales.append(max(math.sqrt(max(total_sq[d] / n - mean * mean, 0.0)), 1e-6))
    return scales


def scaled(fr: list[list[float]], scales: list[float]) -> list[list[float]]:
    return [[x / s for x, s in zip(f, scales)] for f in fr]
