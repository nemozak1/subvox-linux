"""Port of Subvox/Decoding/TemplateClassifier.swift: nearest-neighbour DTW against user-recorded
templates, plus the leave-one-out + label-permutation evaluation that answers "is the signal real?"."""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import dtw, features
from .features import FeatureSet
from .motion import MotionSample


@dataclass
class Template:
    label: str
    samples: list[MotionSample]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def duration(self) -> float:
        return self.samples[-1].t - self.samples[0].t if self.samples else 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "createdAt": self.created_at,
                "samples": [s.to_dict() for s in self.samples]}

    @classmethod
    def from_dict(cls, d: dict) -> "Template":
        return cls(label=d["label"], samples=[MotionSample.from_dict(s) for s in d["samples"]],
                   id=d.get("id", uuid.uuid4().hex), created_at=d.get("createdAt", ""))


@dataclass
class RankedLabel:
    label: str
    distance: float


@dataclass
class Prediction:
    ranked: list[RankedLabel]
    confidence: float  # 0..1 margin between best and second-best class
    accepted: bool

    @property
    def best(self) -> RankedLabel | None:
        return self.ranked[0] if self.ranked else None


def margin(sorted_distances: list[float]) -> float:
    if len(sorted_distances) < 2 or not sorted_distances[1] > 0:
        return 0.0
    return max(0.0, 1 - sorted_distances[0] / sorted_distances[1])


class TemplateClassifier:
    """Small-vocabulary recogniser. Deliberately simple: the question is whether the signal is
    there at all, not how far a big model can push it."""

    def __init__(self, templates: list[Template], feature_set: FeatureSet) -> None:
        if not feature_set:
            raise ValueError("empty feature set")
        if len({t.label for t in templates}) < 2:
            raise ValueError("need templates for at least two words")
        raw = [features.frames(features.trim(t.samples), feature_set) for t in templates]
        self.feature_set = feature_set
        self.scales = features.channel_scales(raw)
        self.entries = [(t.label, features.scaled(fr, self.scales)) for t, fr in zip(templates, raw)]

    @property
    def labels(self) -> list[str]:
        return sorted({label for label, _ in self.entries})

    def rank(self, samples: list[MotionSample]) -> list[RankedLabel]:
        query = features.scaled(features.frames(features.trim(samples), self.feature_set), self.scales)
        best: dict[str, float] = {}
        for label, fr in self.entries:
            d = dtw.distance(query, fr)
            best[label] = min(best.get(label, math.inf), d)
        return sorted((RankedLabel(l, d) for l, d in best.items()), key=lambda r: r.distance)

    def predict(self, samples: list[MotionSample], max_distance: float, min_confidence: float) -> Prediction:
        ranked = self.rank(samples)
        conf = margin([r.distance for r in ranked])
        accepted = bool(ranked) and ranked[0].distance <= max_distance and conf >= min_confidence
        return Prediction(ranked, conf, accepted)


@dataclass
class Evaluation:
    labels: list[str]
    confusion: list[list[int]]  # confusion[actual][predicted]
    accuracy: float
    chance: float  # accuracy of always guessing the most common label
    permutation_mean: float
    permutation_p95: float
    p_value: float
    permutations: int
    template_count: int

    def summary(self) -> str:
        lines = [
            f"{self.template_count} templates, {len(self.labels)} words, features ok",
            f"leave-one-out accuracy: {self.accuracy:.1%}   chance (majority): {self.chance:.1%}",
            f"shuffled-label accuracy: mean {self.permutation_mean:.1%}, 95th pct {self.permutation_p95:.1%}"
            f"   ({self.permutations} permutations)",
            f"p = {self.p_value:.3f}  ->  " + ("SIGNAL (p < 0.05)" if self.p_value < 0.05 else "not distinguishable from chance"),
            "",
            "confusion (rows = actual, cols = predicted):",
        ]
        w = max(len(l) for l in self.labels)
        lines.append(" " * (w + 2) + " ".join(f"{l[:6]:>6}" for l in self.labels))
        for l, row in zip(self.labels, self.confusion):
            lines.append(f"{l:>{w}}  " + " ".join(f"{c:6d}" for c in row))
        return "\n".join(lines)


def evaluate(templates: list[Template], feature_set: FeatureSet, permutations: int = 500,
             seed: int | None = None) -> Evaluation | None:
    """Leave-one-out accuracy plus a label-permutation test."""
    labels = sorted({t.label for t in templates})
    counts = Counter(t.label for t in templates)
    if len(labels) < 2 or any(c < 2 for c in counts.values()) or not feature_set:
        return None

    raw = [features.frames(features.trim(t.samples), feature_set) for t in templates]
    scales = features.channel_scales(raw)
    seqs = [features.scaled(fr, scales) for fr in raw]
    n = len(seqs)

    # Pairwise distances once; every labelling (true or permuted) reuses them.
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw.distance(seqs[i], seqs[j])
            dist[i][j] = dist[j][i] = d

    index = {l: k for k, l in enumerate(labels)}
    truth = [index[t.label] for t in templates]

    def leave_one_out(assignment: list[int]) -> list[int]:
        out = []
        for i in range(n):
            best = [math.inf] * len(labels)
            for j in range(n):
                if j != i:
                    a = assignment[j]
                    if dist[i][j] < best[a]:
                        best[a] = dist[i][j]
            out.append(min(range(len(labels)), key=lambda k: best[k]))
        return out

    def accuracy(pred: list[int], actual: list[int]) -> float:
        return sum(p == a for p, a in zip(pred, actual)) / n

    predicted = leave_one_out(truth)
    confusion = [[0] * len(labels) for _ in labels]
    for p, a in zip(predicted, truth):
        confusion[a][p] += 1
    observed = accuracy(predicted, truth)

    rng = random.Random(seed)
    null = []
    for _ in range(permutations):
        shuffled = truth[:]
        rng.shuffle(shuffled)
        null.append(accuracy(leave_one_out(shuffled), shuffled))
    null.sort()
    at_least = sum(1 for a in null if a >= observed)

    return Evaluation(
        labels=labels,
        confusion=confusion,
        accuracy=observed,
        chance=max(counts.values()) / n,
        permutation_mean=sum(null) / max(permutations, 1),
        permutation_p95=null[int((len(null) - 1) * 0.95)] if null else 0.0,
        p_value=(at_least + 1) / (permutations + 1),
        permutations=permutations,
        template_count=n,
    )
