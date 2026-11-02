#!/usr/bin/env python3
"""Stream head-tracking frames from AirPods / Beats over AAP (L2CAP PSM 0x1001).

Opens the Apple Accessory Protocol channel on an already-connected device, sends the
handshake, asks for notifications and head tracking, then prints every packet. Frames
with opcode 0x17 (head tracking) are logged to a CSV with the raw bytes so the layout
can be worked out offline. At the end it prints which byte offsets changed, which is
the quickest way to find the IMU channels.

Packet bytes come from LibrePods (kavishdevar/librepods): docs/AAP Definitions.md and
android/.../bluetooth/AACPManager.kt.

Usage:
  tools/aap_sniff.py                       # auto-pick the connected device
  tools/aap_sniff.py AA:BB:CC:DD:EE:FF -d 30 -o frames.csv
  tools/aap_sniff.py --service16           # firmware that exposes motion as service 16
"""

import argparse
import csv
import socket
import struct
import subprocess
import sys
import time

PSM = 0x1001
HEADER = bytes.fromhex("04 00 04 00")
HANDSHAKE = bytes.fromhex("00 00 04 00 01 00 02 00 00 00 00 00 00 00 00 00")
SET_FEATURE_FLAGS = HEADER + bytes.fromhex("4D 00 FF 00 00 00 00 00 00 00")
REQUEST_NOTIFICATIONS = HEADER + bytes.fromhex("0F 00 FF FF FF FF")

OP_HEADTRACKING = 0x17
# The 0x17 payload is protobuf: field 1 = sequence, field 8 = { field 1 = service id,
# field 2 = 2, field 3 = bytes }. Powerbeats Pro 2 (fw 0167) announce the "devmotion6"
# service as id 14 (0x0E) and ignore requests for id 16 (0x10), so service 14 is the default.
# LibrePods calls the id-14 packets "primary" and the id-16 ones "alternate".
START_SVC14 = HEADER + bytes.fromhex("17 00 00 00 10 00 10 00 08 A1 02 42 0B 08 0E 10 02 1A 05 01 40 9C 00 00")
STOP_SVC14 = HEADER + bytes.fromhex("17 00 00 00 10 00 11 00 08 7E 10 02 42 0B 08 4E 10 02 1A 05 01 00 00 00 00")
START_SVC16 = HEADER + bytes.fromhex("17 00 00 00 10 00 0F 00 08 73 42 0B 08 10 10 02 1A 05 01 40 9C 00 00")
STOP_SVC16 = HEADER + bytes.fromhex("17 00 00 00 10 00 0F 00 08 75 42 0B 08 10 10 02 1A 05 01 00 00 00 00")

# Offsets LibrePods decodes (little-endian int16): three "orientation" words and two
# acceleration words. Everything else in the ~70-byte frame is still unmapped.
KNOWN_I16 = {"o1": 43, "o2": 45, "o3": 47, "acc_h": 51, "acc_v": 53}


def connected_device() -> str | None:
    out = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True).stdout
    macs = [line.split()[1] for line in out.splitlines() if line.startswith("Device ")]
    return macs[0] if len(macs) == 1 else None


def hexs(b: bytes) -> str:
    return b.hex(" ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mac", nargs="?", help="device address; default: the one connected device")
    ap.add_argument("-d", "--duration", type=float, default=30, help="seconds to capture (default 30)")
    ap.add_argument("-o", "--out", default="frames.csv", help="CSV of 0x17 frames (default frames.csv)")
    ap.add_argument("--service16", action="store_true", help="address motion service 16 instead of 14 (other firmware)")
    ap.add_argument("--no-start", action="store_true", help="only listen; do not request head tracking")
    ap.add_argument("--quiet", action="store_true", help="do not print every 0x17 frame")
    args = ap.parse_args()

    mac = args.mac or connected_device()
    if not mac:
        print("no single connected device; pass the MAC (bluetoothctl devices Connected)", file=sys.stderr)
        return 2
    start, stop = (START_SVC16, STOP_SVC16) if args.service16 else (START_SVC14, STOP_SVC14)

    sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_SEQPACKET, socket.BTPROTO_L2CAP)
    sock.settimeout(5)
    print(f"connecting to {mac} PSM {PSM:#06x} ...")
    sock.connect((mac, PSM))
    print("connected")

    def send(label: str, pkt: bytes) -> None:
        sock.send(pkt)
        print(f"-> {label}: {hexs(pkt)}")

    send("handshake", HANDSHAKE)
    time.sleep(0.2)
    send("feature flags", SET_FEATURE_FLAGS)
    send("notifications", REQUEST_NOTIFICATIONS)
    time.sleep(0.5)
    if not args.no_start:
        send("start head tracking", start)

    t0 = time.monotonic()
    frames = 0
    others = 0
    first_frame_t = None
    minmax: dict[int, tuple[int, int]] = {}
    sock.settimeout(1)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "len", *KNOWN_I16, "hex"])
        try:
            while time.monotonic() - t0 < args.duration:
                try:
                    pkt = sock.recv(1024)
                except socket.timeout:
                    continue
                if not pkt:
                    print("peer closed")
                    break
                t = time.monotonic() - t0
                op = pkt[4] if len(pkt) > 4 and pkt[:4] == HEADER else None
                if op == OP_HEADTRACKING and len(pkt) >= 60:
                    frames += 1
                    first_frame_t = first_frame_t if first_frame_t is not None else t
                    vals = {
                        k: struct.unpack_from("<h", pkt, off)[0] if off + 2 <= len(pkt) else None
                        for k, off in KNOWN_I16.items()
                    }
                    for i, b in enumerate(pkt):
                        lo, hi = minmax.get(i, (b, b))
                        minmax[i] = (min(lo, b), max(hi, b))
                    w.writerow([f"{t:.4f}", len(pkt), *vals.values(), pkt.hex()])
                    if not args.quiet:
                        print(f"{t:7.3f} HT len={len(pkt):3d} " + " ".join(f"{k}={v:6}" for k, v in vals.items()))
                else:
                    others += 1
                    print(f"{t:7.3f} op={op:#04x} len={len(pkt)}: {hexs(pkt)}" if op is not None
                          else f"{t:7.3f} raw len={len(pkt)}: {hexs(pkt)}")
        except KeyboardInterrupt:
            print()
        finally:
            if not args.no_start:
                try:
                    send("stop head tracking", stop)
                except OSError:
                    pass
            sock.close()

    span = (time.monotonic() - t0 - first_frame_t) if first_frame_t is not None else 0
    rate = frames / span if span > 0 else 0
    print(f"\n{frames} head-tracking frames ({rate:.1f} Hz), {others} other packets, CSV: {args.out}")
    if frames:
        moving = [i for i, (lo, hi) in sorted(minmax.items()) if hi - lo > 0]
        print(f"byte offsets that changed ({len(moving)}): {moving}")
        print("constant offsets: " + str([i for i in sorted(minmax) if i not in moving]))
    elif not args.no_start:
        print("no 0x17 frames: this device did not answer the head-tracking request "
              "(try --service16, or DeviceID = bluetooth:004C:0000:0000 in /etc/bluetooth/main.conf)")
    return 0 if frames or args.no_start else 1


if __name__ == "__main__":
    sys.exit(main())
