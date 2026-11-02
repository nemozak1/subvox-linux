# Subvox for Linux

A Linux port of [joeblau/subvox](https://github.com/joeblau/subvox): does the head-motion stream
from Apple earbuds carry enough information to turn **silently mouthed words into text**?

The Mac app reads `CMHeadphoneMotionManager`. Linux has no such API, so this port talks to the
earbuds directly over the Apple Accessory Protocol (AAP, L2CAP PSM 0x1001) and decodes the
motion frames itself. Everything downstream (segmentation, features, DTW matching,
leave-one-out + permutation evaluation) is a line-for-line port of the Swift, in plain Python
with no dependencies.

Verified 2026-11-02 with **Powerbeats Pro 2** (firmware 0167) on BlueZ 5.87: 25 Hz frames, the
same rate the Mac gets.

## Setup

Pair the buds once (case open, hold the system button until the light pulses white):

```sh
tools/pair.sh                 # scans for Beats|AirPods, pairs, trusts, connects
```

Then, with the buds connected:

```sh
python3 -m subvox sensors                # live accel / gyro / attitude / energy
python3 -m subvox train yes no stop go   # Space starts/stops a take; Tab or 1-9 switch word
python3 -m subvox evaluate               # LOO accuracy vs 500 shuffled-label runs, p-value
python3 -m subvox evaluate --features ablation   # all / acc / rot / att / acc+rot
python3 -m subvox decode                 # Space-delimited segments -> words; c calibrates, a auto-detect
python3 -m subvox record silent          # motion.csv session in ~/Documents/Subvox/Sessions/
```

Templates live in `~/.local/share/subvox/dataset.json`. `subvox list` / `subvox forget <word>`.

Tests: `python3 -m unittest tests/test_decoding.py`

## How the motion channel works

`tools/aap_sniff.py` is the exploration tool that found this; `subvox/motion.py` is the
production reader and documents the frame layout.

1. Connect an L2CAP `SOCK_SEQPACKET` socket to the buds on PSM `0x1001` (no root needed).
2. Send the AAP handshake, feature flags and notification request (bytes from LibrePods).
3. Send the opcode `0x17` start packet addressed to motion service **14**. The buds announce
   their sensor as a HID-style `devmotion6` service in a 488-byte descriptor; on this firmware
   its id is 14, and LibrePods' default packet (service 16) is silently ignored.
4. Frames arrive at 25 Hz as protobuf `{1: seq, 2: 1, 7: {1: 14, 3: <58-byte HID report>}}`.

Report layout (offsets into the 58-byte report):

| bytes | field |
|---|---|
| 0 | report id `01` |
| 1–8 | u64 timestamp, ns |
| 9–11 | flags |
| 12–15 | u32 timestamp, µs |
| 16–19 | zero |
| 20–27 | 3 × i16 = vector part of a unit quaternion / 32768 (w = √(1−|v|²)), sensor → gravity-aligned world frame, yaw included; i16 word 3 ≈ 0 |
| 28–33 | 3 × i16, unidentified |
| 34–37 | marker `xx 0E 16 13` |
| 38–43 | 3 × i16 gyroscope, raw |
| 44–49 | 3 × i16 accelerometer, 1 g = 1024 LSB, gravity included, points *down* (along gravity) |
| 50–53 | f32, −1.0 until sensor fusion settles |
| 54–57 | u32 sample counter, +8 per frame |

The reader low-passes the accelerometer for a gravity estimate and subtracts it to get user
acceleration, as CoreMotion does. The gyro is coarse and its integral does not match the quaternion's rotation (scatter
of 5× across moves), so it is only used as a shape feature; the classifier z-scales every channel.
The quaternion was confirmed 2026-11-02 (`subvox calibrate`): q·accel is one fixed world vector in
every head pose (5° scatter, 12° raw), turn axes are exactly ±up, nod axes exactly perpendicular.

## Known limits

Same as the Mac app, plus:

- The classifier still treats the three orientation words as separate channels rather than as the
  quaternion they are.
- Gyro units are approximate (`GYRO_RAD_PER_LSB` in `motion.py`); the raw gyro is not a clean rate.
- One connected device at a time; pass `--mac` if several are connected.
- Terminal UI: Space toggles capture rather than hold-to-capture.

## Head-tracked spatial audio (`subvox spatial`)

Uses the same motion stream to do what Apple's Spatial Audio with head tracking does, on
PipeWire. Verified 2026-11-02 with Powerbeats Pro 2 on PipeWire 1.6.2 (Ubuntu), AAC A2DP.

**Renderer** (`spatial/subvox-spatial.conf`): a filter-chain sink "Spatial (Powerbeats)" whose
FL/FR channels feed two `sofa` spatializers (libmysofa, MIT KEMAR HRTF from the `libmysofa1`
package) at azimuth 30° / 330°, mixed to binaural stereo and sent to the buds
(`bluez_output.<mac>`). It lives in `~/.config/pipewire/filter-chain.conf.d/` and runs in the
user `filter-chain.service`, so it restarts on its own without touching the main daemon:

```sh
tools/install-spatial.sh          # copy config, restart filter-chain, make it the default sink
tools/install-spatial.sh --remove
```

The spatializer controls are live: `pw-cli set-param <sink id> Props '{ params = [ "spFL:Azimuth" 60.0 ] }'`.
SOFA azimuth is counter-clockwise from the front, 0–360. Nothing is installed system-wide.

**Head tracking** (`subvox/spatial.py`): `python3 -m subvox spatial` takes the buds' own fused
attitude quaternion from each frame (`--gyro` integrates the gyro with gravity correction instead)
and every 100 ms asks where the two room-fixed ±30° speakers are seen from the head, pushing their
Azimuth/Elevation to the sink through a long-lived `pw-cli`. The reference frame slides back to the current pose while the head
is still (`--recentre 6` s time constant), as Apple's does; `c` or `python3 -m subvox recentre` from
another shell (SIGUSR1) makes where you look now the front at once. Regenerating
the HRIRs at 10 Hz costs the filter-chain process ~0.6 % CPU. `--demo` sweeps yaw ±60° without
the buds. A2DP adds ~150–250 ms, so it lags Apple's.

**Calibration** (`subvox/calibrate.py`): `python3 -m subvox calibrate` prompts 13 timed moves
(still, left/right 90°, up/down, tilt) and records raw frames to
`~/.local/share/subvox/calibration/<timestamp>.csv`, then verifies the quaternion, derives the
head's up/left/forward axes in the sensor frame (up from gravity, left from the quaternion change
when nodding), the world up vector, and a gyro scale as a fallback. It writes
`~/.local/share/subvox/head-calibration.json`, which `spatial` loads. `--analyze <csv> [--save]`
re-runs the analysis on a recording. Uncalibrated, `spatial` takes "up" from the first second at
rest and gets yaw right but mixes pitch and roll.
