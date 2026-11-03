"""Terminal front end: the Sensors / Train / Decode screens of the Mac app, in one process.

  subvox sensors                      live sample rate, accel, gyro, energy
  subvox train <word> [...]           Space to start/stop a take, Tab or 1-9 to switch word, q quit
  subvox evaluate [--features acc,rot,att] [--permutations 500]
  subvox decode [--features ...] [--max-distance 1.0] [--min-confidence 0.1]
  subvox record <condition>           motion.csv session like the Mac app's Record Session
  subvox calibrate [--analyze file]   scripted head moves -> head-calibration.json for spatial
  subvox spatial [--demo]             head-tracked spatial audio on the "Spatial (Powerbeats)" sink
  subvox recentre                     tell a running `spatial` that where you look now is the front
  subvox ctl on|off|toggle|status|set|linger|settings|install   desktop controls (see ctl.py)
  subvox list | subvox forget <word>
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

from . import store
from .capture import CaptureController
from .classifier import Template, TemplateClassifier, evaluate
from .features import FeatureSet, energy
from .motion import HeadphoneMotion, MotionSample, connected_device
from . import calibrate as calib
from . import spatial as sp
from .settings import Settings, write_status


class Keys:
    """Raw-mode single-key reader usable from a poll loop."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.enabled = sys.stdin.isatty()
        self.saved = None

    def __enter__(self) -> "Keys":
        if self.enabled:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        r, _, _ = select.select([self.fd], [], [], 0)
        return sys.stdin.read(1) if r else None


def motion_thread(mac: str, q: "queue.Queue[MotionSample | None]", stop: threading.Event) -> HeadphoneMotion:
    motion = HeadphoneMotion(mac)
    motion.start()

    def run() -> None:
        try:
            for s in motion.samples():
                q.put(s)
                if stop.is_set():
                    break
        except OSError:
            pass                      # the buds reset AAP when A2DP reconnects; the caller reopens
        finally:
            motion.stop()
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return motion


def resolve_mac(arg: str | None) -> str:
    mac = arg or connected_device()
    if not mac:
        sys.exit("no single connected Bluetooth device; pass --mac")
    return mac


def status_line(text: str) -> None:
    sys.stdout.write("\r\x1b[2K" + text)
    sys.stdout.flush()


# ---------------------------------------------------------------- commands

def cmd_sensors(a: argparse.Namespace) -> None:
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    motion = motion_thread(resolve_mac(a.mac), q, stop)
    print("streaming; Ctrl-C to stop")
    cap = CaptureController()
    try:
        while True:
            s = q.get()
            if s is None:
                print("\nstream ended"); return
            cap.ingest(s)
            status_line(
                f"{motion.sample_rate:5.1f} Hz  acc g ({s.ax:+.3f} {s.ay:+.3f} {s.az:+.3f})  "
                f"gyro ({s.rx:+.3f} {s.ry:+.3f} {s.rz:+.3f})  att ({s.pitch:+.2f} {s.roll:+.2f} {s.yaw:+.2f})  "
                f"energy {cap.energy:.3f}  n={motion.sample_count}")
    except KeyboardInterrupt:
        stop.set()
        print()


def cmd_record(a: argparse.Namespace) -> None:
    folder = store.SESSIONS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{a.condition}"
    folder.mkdir(parents=True, exist_ok=True)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    motion = motion_thread(resolve_mac(a.mac), q, stop)
    t0 = None
    n = 0
    print(f"recording to {folder}; Ctrl-C to stop")
    with open(folder / "motion.csv", "w") as f:
        f.write(MotionSample.CSV_HEADER + "\n")
        try:
            while True:
                s = q.get()
                if s is None:
                    break
                t0 = s.t if t0 is None else t0
                f.write(s.csv_row(t0) + "\n")
                n += 1
                if n % 25 == 0:
                    status_line(f"{s.t - t0:7.1f} s  {motion.sample_rate:.1f} Hz  {n} samples")
        except KeyboardInterrupt:
            stop.set()
    (folder / "meta.json").write_text(json.dumps({"condition": a.condition, "samples": n, "device": motion.mac}, indent=2))
    print(f"\nsaved {n} samples")


def cmd_train(a: argparse.Namespace) -> None:
    templates = store.load()
    words = a.words
    if not words:
        words = sorted({t.label for t in templates})
    if not words:
        sys.exit("give at least one word to train")
    current = 0
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    motion = motion_thread(resolve_mac(a.mac), q, stop)
    cap = CaptureController()

    def on_segment(seg: list[MotionSample]) -> None:
        templates.append(Template(label=words[current], samples=seg))
        store.save(templates)
        count = sum(1 for t in templates if t.label == words[current])
        print(f"\r\x1b[2K  take saved: '{words[current]}'  {seg[-1].t - seg[0].t:.2f} s, {len(seg)} frames"
              f"  ({count} takes for this word, {len(templates)} total)")

    cap.on_segment = on_segment
    print("Train. Space: start/stop a take while mouthing the word.  Tab or 1-9: switch word.  q: quit")
    print("words: " + "  ".join(f"[{i + 1}] {w}" for i, w in enumerate(words)))
    with Keys() as keys:
        try:
            while True:
                try:
                    s = q.get(timeout=0.05)
                except queue.Empty:
                    s = False
                if s is None:
                    print("\nstream ended"); break
                if s:
                    cap.ingest(s)
                k = keys.poll()
                if k == "q":
                    break
                elif k == " ":
                    cap.end() if cap.is_capturing else cap.begin()
                elif k == "\t":
                    current = (current + 1) % len(words)
                elif k and k.isdigit() and 1 <= int(k) <= len(words):
                    current = int(k) - 1
                count = sum(1 for t in templates if t.label == words[current])
                state = "● RECORDING" if cap.is_capturing else "  ready    "
                status_line(f"{state}  word: {words[current]} ({count} takes)  energy {cap.energy:.3f}  {motion.sample_rate:.0f} Hz")
        except KeyboardInterrupt:
            pass
    stop.set()
    print()


def cmd_evaluate(a: argparse.Namespace) -> None:
    templates = store.load()
    fs = FeatureSet.parse(a.features)
    sets = [fs] if a.features != "ablation" else [
        FeatureSet.all(), FeatureSet.ACCELERATION, FeatureSet.ROTATION, FeatureSet.ATTITUDE,
        FeatureSet.ACCELERATION | FeatureSet.ROTATION]
    for s in sets:
        print(f"\n=== features: {s.label()}")
        ev = evaluate(templates, s, permutations=a.permutations)
        if ev is None:
            print("need at least 2 words with at least 2 takes each"); return
        print(ev.summary())


def cmd_decode(a: argparse.Namespace) -> None:
    templates = store.load()
    fs = FeatureSet.parse(a.features)
    try:
        clf = TemplateClassifier(templates, fs)
    except ValueError as e:
        sys.exit(str(e))
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    motion = motion_thread(resolve_mac(a.mac), q, stop)
    cap = CaptureController()
    transcript: list[str] = []

    def on_segment(seg: list[MotionSample]) -> None:
        p = clf.predict(seg, a.max_distance, a.min_confidence)
        ranked = "  ".join(f"{r.label}:{r.distance:.2f}" for r in p.ranked[:3])
        if p.accepted and p.best:
            transcript.append(p.best.label)
            print(f"\r\x1b[2K  {p.best.label:<12} conf {p.confidence:.2f}   {ranked}")
        else:
            print(f"\r\x1b[2K  (rejected)   conf {p.confidence:.2f}   {ranked}")

    cap.on_segment = on_segment
    print(f"Decode with {clf.labels}. Space: start/stop a segment.  c: calibrate noise floor (hold still 2 s).  "
          f"a: toggle auto-detect.  q: quit")
    with Keys() as keys:
        try:
            while True:
                try:
                    s = q.get(timeout=0.05)
                except queue.Empty:
                    s = False
                if s is None:
                    print("\nstream ended"); break
                if s:
                    cap.ingest(s)
                k = keys.poll()
                if k == "q":
                    break
                elif k == " ":
                    cap.end() if cap.is_capturing else cap.begin()
                elif k == "c":
                    cap.calibrate()
                elif k == "a":
                    cap.set_auto(not cap.auto_enabled)
                mode = "auto" if cap.auto_enabled else "hold"
                thr = f"thr {cap.threshold:.3f}" if cap.threshold else "uncalibrated"
                state = "● CAPTURING" if cap.is_capturing else ("calibrating" if cap.is_calibrating else "  ready    ")
                status_line(f"{state}  {mode}  {thr}  energy {cap.energy:.3f}  {motion.sample_rate:.0f} Hz  | "
                            + " ".join(transcript[-8:]))
        except KeyboardInterrupt:
            pass
    stop.set()
    print("\ntranscript:", " ".join(transcript))


def cmd_calibrate(a: argparse.Namespace) -> None:
    if a.analyze:
        rows = calib.read_recording(Path(a.analyze))
    else:
        q: queue.Queue = queue.Queue()
        stop = threading.Event()
        motion = motion_thread(resolve_mac(a.mac), q, stop)
        total = sum(sec for _, _, sec in calib.PHASES)
        print(f"{len(calib.PHASES)} prompts, {total} s. Wear the buds, face where the music should come from.")
        input("Press Enter to start ...")
        while not q.empty():      # drop everything streamed before the first prompt
            q.get_nowait()
        rows: list[dict] = []
        phase = "start"
        t_end = time.monotonic()
        try:
            for label, text, secs in calib.PHASES:
                print(f"\n>> {text}")
                t_end = time.monotonic() + secs
                phase = label
                while time.monotonic() < t_end:
                    try:
                        s = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if s is None:
                        sys.exit("stream ended")
                    rows.append({"t": f"{s.t:.4f}", "phase": phase, "counter": s.counter,
                                 "o0": s.raw_orient[0], "o1": s.raw_orient[1], "o2": s.raw_orient[2], "o3": s.raw_orient[3],
                                 "gx": s.raw_gyro[0], "gy": s.raw_gyro[1], "gz": s.raw_gyro[2],
                                 "ax": s.raw_accel[0], "ay": s.raw_accel[1], "az": s.raw_accel[2]})
                    status_line(f"   {phase:11} {t_end - time.monotonic():4.1f} s  {motion.sample_rate:4.1f} Hz  "
                                f"orient {s.raw_orient}  gyro {s.raw_gyro}")
        finally:
            stop.set()
        print()
        rec = calib.RECORDINGS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
        calib.write_recording(rows, rec)
        print(f"recorded {len(rows)} frames to {rec}")
    cal, report = calib.analyze(rows)
    print(report)
    if a.analyze and not a.save:
        print("\n(--save to write head-calibration.json)")
        return
    cal.save()
    print(f"\nsaved {calib.CALIBRATION_PATH}")


def cmd_spatial(a: argparse.Namespace) -> None:
    node = sp.find_node()
    if node is None:
        sys.exit("sink 'subvox_spatial' not found; run tools/install-spatial.sh")
    settings = Settings()
    for key in ("rate", "spread", "smooth", "lead", "deadband", "linger"):
        val = getattr(a, key, None)
        if val is not None and key != "smooth":
            settings.set(key, val)
    if a.no_elevation:
        settings.set("elevation", False)
    if not settings.path.exists():
        settings.save()
    control = sp.SpatializerControl(node)
    renderer = sp.StageRenderer(control, smooth_tau=a.smooth if a.smooth is not None else 0.04)

    def apply_settings() -> float:
        renderer.spread = settings["spread"]
        renderer.lead = settings["lead"]
        renderer.min_change = settings["deadband"]
        renderer.elevation = settings["elevation"]
        return 1.0 / settings["rate"]

    period = apply_settings()
    if a.demo:
        tracker = sp.DemoTracker()
        cpu0, t0 = sp.filter_chain_cpu(), time.monotonic()
        print(f"demo: sweeping yaw ±60° into sink {node} at {1 / period:.0f} Hz; Ctrl-C to stop")
        try:
            while True:
                pose = tracker.tick()
                renderer.update(tracker, time.monotonic())
                cpu = sp.filter_chain_cpu()
                load = (cpu - cpu0) / (time.monotonic() - t0) * 100 if cpu is not None and cpu0 is not None else float("nan")
                status_line(f"yaw {pose.yaw:+6.1f}°  FL {renderer.state['spFL'][0]:6.1f}° FR {renderer.state['spFR'][0]:6.1f}°  "
                            f"updates {renderer.updates}  filter-chain CPU {load:4.1f} %")
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        finally:
            renderer.reset(); control.close(); print()
        return

    cal = calib.HeadCalibration.load()
    src = ("calibrated" if cal.up else "uncalibrated: first second at rest defines up, yaw only is right") + \
          (", gyro integration" if a.gyro else ", buds' attitude quaternion")
    tracker = sp.HeadTracker(cal, recentre_tau=settings["linger"], use_quaternion=not a.gyro)
    print(f"spatial: sink {node}, {src}; keys: c recentre, q quit; `subvox recentre` from another shell")
    signal.signal(signal.SIGUSR1, lambda *_: tracker.recentre())
    import os
    calib.CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPATIAL_PID.write_text(str(os.getpid()))
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    motion = None
    stream_state = "connecting"
    last_sample = time.monotonic()
    retry_at = 0.0
    next_tick = time.monotonic()
    next_check = time.monotonic() + 10
    next_status = 0.0
    pose = None
    try:
        with Keys() as keys:
            while True:
                now = time.monotonic()
                if motion is None and now >= retry_at:
                    # (Re)open the motion stream. The buds reset AAP when A2DP reconnects.
                    mac = a.mac or connected_device()
                    try:
                        if not mac:
                            raise OSError("buds not connected")
                        motion = motion_thread(mac, q, stop)
                        tracker.ref_set = False      # take a fresh front once still
                        stream_state = "open"
                        last_sample = time.monotonic()
                        status_line("motion stream open")
                    except OSError as e:
                        stream_state = "reconnecting"
                        status_line(f"motion: {e}; retrying in 3 s")
                        retry_at = now + 3
                try:
                    s = q.get(timeout=0.05)
                    if s is None:
                        status_line("motion stream ended; reconnecting")
                        motion = None
                        stream_state = "reconnecting"
                        retry_at = time.monotonic() + 2
                        renderer.reset()
                        continue
                    pose = tracker.update(s) or pose
                    last_sample = time.monotonic()
                except queue.Empty:
                    pass
                if motion is not None and time.monotonic() - last_sample > 2.0:
                    # Socket still open but no frames: the buds stop the motion service when they
                    # leave the ears and need a fresh START. Closing the socket ends the reader
                    # thread, which queues None, and the reconnect path above reopens it.
                    status_line("no motion frames for 2 s; restarting the stream")
                    motion.stop()
                    motion = None
                    stream_state = "reconnecting"
                    retry_at = time.monotonic() + 1
                    renderer.reset()
                k = keys.poll()
                if k == "q":
                    break
                if k == "c":
                    tracker.recentre()
                now = time.monotonic()
                if now >= next_status:
                    next_status = now + 0.5
                    if settings.reload():
                        period = apply_settings()
                        tracker.recentre_tau = settings["linger"]
                    write_status(stream=stream_state, hz=motion.sample_rate if motion else 0.0,
                                 yaw=pose.yaw if pose else 0.0, pitch=pose.pitch if pose else 0.0,
                                 roll=pose.roll if pose else 0.0, still=pose.still if pose else False,
                                 rate=pose.rate if pose else 0.0, sent=renderer.updates)
                if now >= next_check:
                    # the sink node is recreated with a new id if the filter-chain restarts
                    next_check = now + 10
                    nid = sp.find_node()
                    if nid is not None and nid != control.node_id:
                        control.node_id = nid
                        renderer.last_sent = {k: (-999.0, -999.0) for k in sp.SPEAKERS}
                if now >= next_tick and tracker.ready and motion is not None:
                    next_tick = now + period
                    renderer.update(tracker, now)
                    if pose:
                        st = renderer.state
                        status_line(f"{motion.sample_rate:4.1f} Hz  yaw {pose.yaw:+6.1f}° pitch {pose.pitch:+5.1f}° roll {pose.roll:+5.1f}°  "
                                    f"{'still ' if pose.still else 'moving'} {pose.rate:5.1f}°/s  "
                                    f"FL {st['spFL'][0]:5.1f}°/{st['spFL'][1]:+5.1f}°  FR {st['spFR'][0]:5.1f}°/{st['spFR'][1]:+5.1f}°  "
                                    f"sent {renderer.updates}" + (f"  gyro {tracker.scale:.2e} rad/LSB (n={len(tracker.scale_estimates)})" if a.gyro else ""))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        renderer.reset()
        control.close()
        write_status(stream="stopped")
        SPATIAL_PID.unlink(missing_ok=True)
        print()


SPATIAL_PID = calib.CALIBRATION_PATH.parent / "spatial.pid"


def cmd_recentre(a: argparse.Namespace) -> None:
    import os
    try:
        os.kill(int(SPATIAL_PID.read_text()), signal.SIGUSR1)
        print("recentred")
    except (OSError, ValueError):
        sys.exit("no running `subvox spatial`")


def cmd_ctl(a: argparse.Namespace) -> None:
    from . import ctl
    ctl.main(a.args)


def cmd_list(a: argparse.Namespace) -> None:
    templates = store.load()
    if not templates:
        print(f"no templates yet ({store.DEFAULT_PATH})"); return
    by: dict[str, list[Template]] = {}
    for t in templates:
        by.setdefault(t.label, []).append(t)
    for label, ts in sorted(by.items()):
        d = [t.duration for t in ts]
        print(f"{label:<14} {len(ts):3d} takes   duration {min(d):.2f}-{max(d):.2f} s")
    print(f"{len(templates)} templates in {store.DEFAULT_PATH}")


def cmd_forget(a: argparse.Namespace) -> None:
    templates = store.load()
    keep = [t for t in templates if t.label != a.word]
    store.save(keep)
    print(f"removed {len(templates) - len(keep)} takes of '{a.word}'")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="subvox", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mac", help="device address (default: the single connected device)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sensors").set_defaults(fn=cmd_sensors)
    p = sub.add_parser("record"); p.add_argument("condition"); p.set_defaults(fn=cmd_record)
    p = sub.add_parser("train"); p.add_argument("words", nargs="*"); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("evaluate"); p.add_argument("--features", default="all", help="all | ablation | acc,rot,att")
    p.add_argument("--permutations", type=int, default=500); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("decode"); p.add_argument("--features", default="all")
    p.add_argument("--max-distance", type=float, default=1.0); p.add_argument("--min-confidence", type=float, default=0.1)
    p.set_defaults(fn=cmd_decode)
    p = sub.add_parser("calibrate"); p.add_argument("--analyze", metavar="CSV", help="re-analyse a recording instead of recording")
    p.add_argument("--save", action="store_true", help="with --analyze: write head-calibration.json"); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("spatial"); p.add_argument("--demo", action="store_true", help="sweep yaw without the buds (load test)")
    p.add_argument("--rate", type=float, help="max control updates per second (each one rebuilds the HRIRs); default from spatial.json")
    p.add_argument("--spread", type=float, help="speaker half-angle, degrees")
    p.add_argument("--smooth", type=float, help="smoothing time constant, s (default 0.04)")
    p.add_argument("--lead", type=float, help="aim at the head pose this far ahead, s (A2DP ~0.21 + quantum 0.02; 0 = off)")
    p.add_argument("--linger", "--recentre", dest="linger", type=float, help="seconds for the stage to drift back when still (0 = never)")
    p.add_argument("--gyro", action="store_true", help="integrate the gyro instead of using the attitude quaternion")
    p.add_argument("--deadband", type=float, help="only send when a speaker moved this many degrees")
    p.add_argument("--no-elevation", action="store_true"); p.set_defaults(fn=cmd_spatial)
    sub.add_parser("recentre").set_defaults(fn=cmd_recentre)
    p = sub.add_parser("ctl", add_help=False); p.add_argument("args", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_ctl)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("forget"); p.add_argument("word"); p.set_defaults(fn=cmd_forget)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
