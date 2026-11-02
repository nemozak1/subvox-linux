"""Port of Subvox/Decoding/CaptureController.swift: cuts the motion stream into word-sized
segments, either while a key is held (begin/end) or automatically when motion energy rises
above a calibrated noise floor."""

from __future__ import annotations

from typing import Callable

from .features import energy
from .motion import MotionSample

CALIBRATION_SECONDS = 2.0
PREROLL_SECONDS = 0.3
HANGOVER_SECONDS = 0.35
MAX_AUTO_SECONDS = 3.0
MAX_HOLD_SECONDS = 6.0


class CaptureController:
    def __init__(self) -> None:
        self.auto_enabled = False
        self.trigger_ratio = 3.0
        self.is_capturing = False
        self.is_calibrating = False
        self.energy = 0.0
        self.noise_floor: float | None = None
        self.on_segment: Callable[[list[MotionSample]], None] | None = None
        self._manual = False
        self._buffer: list[MotionSample] = []
        self._preroll: list[MotionSample] = []
        self._calibration: list[float] = []
        self._calibration_end: float | None = None
        self._quiet_since: float | None = None

    @property
    def threshold(self) -> float | None:
        return None if self.noise_floor is None else self.noise_floor * self.trigger_ratio

    def begin(self) -> None:
        if self.is_capturing:
            return
        self._manual = True
        self.is_capturing = True
        self._buffer = list(self._preroll)

    def end(self) -> None:
        if self.is_capturing and self._manual:
            self._finish()

    def set_auto(self, enabled: bool) -> None:
        self.auto_enabled = enabled
        if not enabled and self.is_capturing and not self._manual:
            self._finish()

    def calibrate(self) -> None:
        self._calibration = []
        self._calibration_end = None
        self.is_calibrating = True

    def ingest(self, s: MotionSample) -> None:
        e = energy(s)
        self.energy = self.energy * 0.6 + e * 0.4

        if self.is_calibrating:
            end = self._calibration_end if self._calibration_end is not None else s.t + CALIBRATION_SECONDS
            self._calibration_end = end
            self._calibration.append(self.energy)
            if s.t >= end:
                ordered = sorted(self._calibration)
                self.noise_floor = max(ordered[len(ordered) // 2], 1e-4)
                self.is_calibrating = False

        self._preroll.append(s)
        self._preroll = [p for p in self._preroll if p.t >= s.t - PREROLL_SECONDS]

        if self.is_capturing:
            self._buffer.append(s)
            duration = s.t - self._buffer[0].t
            if self._manual:
                if duration > MAX_HOLD_SECONDS:
                    self._finish()
            elif (thr := self.threshold) is not None:
                if self.energy < thr * 0.7:
                    since = self._quiet_since if self._quiet_since is not None else s.t
                    self._quiet_since = since
                    if s.t - since > HANGOVER_SECONDS:
                        self._finish()
                else:
                    self._quiet_since = None
                if self.is_capturing and duration > MAX_AUTO_SECONDS:
                    self._finish()
        elif self.auto_enabled and not self.is_calibrating and (thr := self.threshold) is not None and self.energy > thr:
            self._manual = False
            self.is_capturing = True
            self._quiet_since = None
            self._buffer = list(self._preroll)

    def _finish(self) -> None:
        segment = self._buffer
        self._buffer = []
        self.is_capturing = False
        self._manual = False
        self._quiet_since = None
        if len(segment) >= 4 and segment[-1].t - segment[0].t >= 0.15 and self.on_segment:
            self.on_segment(segment)
