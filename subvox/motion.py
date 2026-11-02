"""Head-motion stream from AirPods / Beats over AAP (L2CAP PSM 0x1001), the Linux stand-in
for CMHeadphoneMotionManager.

Protocol (worked out on Powerbeats Pro 2, firmware 0167, 2026-11-02; packet bytes from
LibrePods): after the AAP handshake, opcode 0x17 carries protobuf
`{1: seq, 8: {1: service_id, 2: 2, 3: bytes}}`. The buds announce a HID-like "devmotion6"
service (id 14 here). Asking it to start yields 25 Hz frames `{1: seq, 2: 1, 7: {1: 14,
3: <58-byte HID input report>}}` with this report layout:

  0      report id 0x01
  1-8    u64 timestamp, nanoseconds (device clock)
  9-11   flags
  12-15  u32 timestamp, microseconds
  16-19  zero
  20-27  4 x i16 orientation words (slow attitude; scale unknown)
  28-33  3 x i16 unknown (one fast axis, one slow ramp)
  34-37  marker xx 0E 16 13
  38-43  3 x i16 gyroscope (raw; ~0 at rest)
  44-49  3 x i16 accelerometer, 1 g = 1024 LSB, gravity included
  50-53  f32, -1.0 until fusion settles
  54-57  u32 sample counter, +8 per frame (200 Hz sensor decimated to 25 Hz)
"""

from __future__ import annotations

import math
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

PSM = 0x1001
HEADER = bytes.fromhex("04 00 04 00")
HANDSHAKE = bytes.fromhex("00 00 04 00 01 00 02 00 00 00 00 00 00 00 00 00")
SET_FEATURE_FLAGS = HEADER + bytes.fromhex("4D 00 FF 00 00 00 00 00 00 00")
REQUEST_NOTIFICATIONS = HEADER + bytes.fromhex("0F 00 FF FF FF FF")
OP_HEADTRACKING = 0x17

START = {
    14: HEADER + bytes.fromhex("17 00 00 00 10 00 10 00 08 A1 02 42 0B 08 0E 10 02 1A 05 01 40 9C 00 00"),
    16: HEADER + bytes.fromhex("17 00 00 00 10 00 0F 00 08 73 42 0B 08 10 10 02 1A 05 01 40 9C 00 00"),
}
STOP = {
    14: HEADER + bytes.fromhex("17 00 00 00 10 00 11 00 08 7E 10 02 42 0B 08 4E 10 02 1A 05 01 00 00 00 00"),
    16: HEADER + bytes.fromhex("17 00 00 00 10 00 0F 00 08 75 42 0B 08 10 10 02 1A 05 01 00 00 00 00"),
}

ACCEL_LSB_PER_G = 1024.0
# Gyro LSB scale is unknown. ~1e-3 rad/s per LSB puts a brisk nod near 1 rad/s, which is
# where the Swift app's energy weighting (rotation + 5 * accel) expects it.
GYRO_RAD_PER_LSB = 1e-3
ORIENT_RAD_PER_LSB = math.pi / 32768.0
GRAVITY_ALPHA = 0.1  # EMA low-pass for the gravity estimate at 25 Hz (~0.4 s)


@dataclass
class MotionSample:
    """One processed reading; the same channels as the Swift `MotionSample`.

    `t` is seconds on the device clock. ax..az is user acceleration in g (gravity removed);
    gx..gz the gravity estimate in g; rx..rz rotation rate (rad/s, approximate scale);
    pitch/roll/yaw the three orientation words in pseudo-radians (scale unknown, so only
    differences within a segment are meaningful).
    """

    t: float
    ax: float; ay: float; az: float
    rx: float; ry: float; rz: float
    pitch: float; roll: float; yaw: float
    gx: float; gy: float; gz: float
    raw_accel: tuple[int, int, int] = field(default=(0, 0, 0), repr=False)
    raw_gyro: tuple[int, int, int] = field(default=(0, 0, 0), repr=False)
    raw_orient: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), repr=False)
    counter: int = 0

    CSV_HEADER = "t,ax,ay,az,rx,ry,rz,pitch,roll,yaw,gx,gy,gz,qw,qx,qy,qz"

    def csv_row(self, t0: float) -> str:
        vals = [self.t - t0, self.ax, self.ay, self.az, self.rx, self.ry, self.rz,
                self.pitch, self.roll, self.yaw, self.gx, self.gy, self.gz, 1.0, 0.0, 0.0, 0.0]
        return ",".join(f"{v:.6f}" for v in vals)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("t", "ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw", "gx", "gy", "gz")}

    @classmethod
    def from_dict(cls, d: dict) -> "MotionSample":
        return cls(**{k: float(d[k]) for k in
                      ("t", "ax", "ay", "az", "rx", "ry", "rz", "pitch", "roll", "yaw", "gx", "gy", "gz")})


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return v, i


def _protobuf(b: bytes) -> dict[int, list]:
    i, out = 0, {}
    while i < len(b):
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            n, i = _varint(b, i); v = b[i:i + n]; i += n
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.setdefault(f, []).append(v)
    return out


def parse_report(pkt: bytes) -> bytes | None:
    """Return the 58-byte HID input report from a 0x17 AAP packet, or None if it is not one."""
    if len(pkt) < 12 or pkt[:4] != HEADER or pkt[4] != OP_HEADTRACKING:
        return None
    n = struct.unpack_from("<H", pkt, 10)[0]
    try:
        top = _protobuf(pkt[12:12 + n])
        inner = _protobuf(top[7][0])
        report = inner[3][0]
    except (KeyError, IndexError, ValueError):
        return None
    return report if len(report) >= 58 and report[0] == 0x01 else None


class MotionDecoder:
    """Turns raw HID reports into MotionSamples, tracking a gravity estimate."""

    def __init__(self) -> None:
        self.gravity: list[float] | None = None

    def decode(self, report: bytes) -> MotionSample:
        t_ns = struct.unpack_from("<Q", report, 1)[0]
        orient = struct.unpack_from("<4h", report, 20)
        gyro = struct.unpack_from("<3h", report, 38)
        accel = struct.unpack_from("<3h", report, 44)
        counter = struct.unpack_from("<I", report, 54)[0]

        a = [v / ACCEL_LSB_PER_G for v in accel]
        if self.gravity is None:
            self.gravity = list(a)
        else:
            self.gravity = [g + GRAVITY_ALPHA * (x - g) for g, x in zip(self.gravity, a)]
        g = self.gravity
        return MotionSample(
            t=t_ns / 1e9,
            ax=a[0] - g[0], ay=a[1] - g[1], az=a[2] - g[2],
            rx=gyro[0] * GYRO_RAD_PER_LSB, ry=gyro[1] * GYRO_RAD_PER_LSB, rz=gyro[2] * GYRO_RAD_PER_LSB,
            pitch=orient[0] * ORIENT_RAD_PER_LSB, roll=orient[1] * ORIENT_RAD_PER_LSB, yaw=orient[2] * ORIENT_RAD_PER_LSB,
            gx=g[0], gy=g[1], gz=g[2],
            raw_accel=accel, raw_gyro=gyro, raw_orient=orient, counter=counter,
        )


class HeadphoneMotion:
    """Opens AAP on a connected device and streams MotionSamples.

    Usage:
        with HeadphoneMotion(mac) as motion:
            for sample in motion.samples():
                ...
    """

    def __init__(self, mac: str, service_id: int = 14, connect_timeout: float = 5.0) -> None:
        self.mac = mac
        self.service_id = service_id
        self.connect_timeout = connect_timeout
        self.sock: socket.socket | None = None
        self.decoder = MotionDecoder()
        self.sample_count = 0
        self.other_packets: list[bytes] = []
        self.on_other: Callable[[bytes], None] | None = None
        self._rate_window: list[float] = []
        self.sample_rate = 0.0

    def __enter__(self) -> "HeadphoneMotion":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
        s.settimeout(self.connect_timeout)
        s.connect((self.mac, PSM))
        s.send(HANDSHAKE)
        time.sleep(0.2)
        s.send(SET_FEATURE_FLAGS)
        s.send(REQUEST_NOTIFICATIONS)
        time.sleep(0.3)
        s.send(START[self.service_id])
        s.settimeout(1.0)
        self.sock = s

    def stop(self) -> None:
        if self.sock is None:
            return
        try:
            self.sock.send(STOP[self.service_id])
        except OSError:
            pass
        self.sock.close()
        self.sock = None

    def samples(self) -> Iterator[MotionSample]:
        """Yield samples until the peer closes the channel. Blocks ≤1 s between checks."""
        assert self.sock is not None, "call start() first"
        while True:
            try:
                pkt = self.sock.recv(1024)
            except socket.timeout:
                continue
            if not pkt:
                return
            report = parse_report(pkt)
            if report is None:
                if self.on_other:
                    self.on_other(pkt)
                continue
            sample = self.decoder.decode(report)
            self.sample_count += 1
            self._rate_window.append(sample.t)
            self._rate_window = [t for t in self._rate_window if t >= sample.t - 1]
            if len(self._rate_window) > 1:
                self.sample_rate = (len(self._rate_window) - 1) / (self._rate_window[-1] - self._rate_window[0])
            yield sample


def connected_device() -> str | None:
    """MAC of the single connected Bluetooth device, via bluetoothctl."""
    import subprocess
    out = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True).stdout
    macs = [line.split()[1] for line in out.splitlines() if line.startswith("Device ")]
    return macs[0] if len(macs) == 1 else None
