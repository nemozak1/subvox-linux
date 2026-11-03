"""`subvox ctl`: switch head-tracked spatial audio on and off and edit its settings.

  subvox ctl on|off|toggle      start/stop the daemon (user service), make the spatial sink or
                                the buds the default output and move playing streams across
  subvox ctl status [--json]    settings + daemon state + where the audio goes
  subvox ctl set <key> <value>  linger, lead, deadband, spread, elevation, rate (daemon reloads)
  subvox ctl linger +1|-1|<s>   nudge or set the linger time
  subvox ctl recentre           where you look now is the front
  subvox ctl settings           open (or close) the settings popup
  subvox ctl install            write the user service and ~/.local/bin/subvox, print the Waybar module

"off" is plain stereo straight to the buds over A2DP: nothing between the player and the codec.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .settings import SETTINGS_PATH, Settings, read_status

SINK = "subvox_spatial"
SERVICE = "subvox-spatial"
REPO = Path(__file__).resolve().parent.parent
UNIT_PATH = Path.home() / ".config/systemd/user" / f"{SERVICE}.service"
WRAPPER = Path.home() / ".local/bin/subvox"
POPUP_PID = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "subvox/settings-popup.pid"
PID_FILE = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "subvox/spatial.pid"


# ---------------------------------------------------------------- pipewire helpers

def pw_dump() -> list[dict]:
    try:
        return json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def graph() -> dict:
    """Spatial sink, buds sink, output streams and which sink each stream feeds."""
    objs = pw_dump()
    nodes = {o["id"]: o for o in objs if o.get("type") == "PipeWire:Interface:Node"}
    props = {i: n.get("info", {}).get("props", {}) for i, n in nodes.items()}
    spatial = next((i for i, p in props.items() if p.get("node.name") == SINK), None)
    buds = next((i for i, p in props.items() if str(p.get("node.name", "")).startswith("bluez_output.")
                 and p.get("media.class") == "Audio/Sink"), None)
    streams = [i for i, p in props.items() if p.get("media.class") == "Stream/Output/Audio"
               and not str(p.get("node.name", "")).startswith(SINK)]
    feeds: dict[int, set[int]] = {s: set() for s in streams}
    for o in objs:
        if o.get("type") == "PipeWire:Interface:Link":
            inf = o.get("info", {})
            out_node, in_node = inf.get("output-node-id"), inf.get("input-node-id")
            if out_node in feeds:
                feeds[out_node].add(in_node)
    default = None
    try:
        out = subprocess.run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "node.name" in line:
                default = line.split('"')[1]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    codec = props.get(buds, {}).get("api.bluez5.codec") if buds else None
    return {"spatial": spatial, "buds": buds, "streams": streams, "feeds": feeds, "default": default,
            "buds_name": props.get(buds, {}).get("node.name") if buds else None, "codec": codec,
            "stream_names": {s: props[s].get("node.name") for s in streams}}


def set_default(node_id: int) -> None:
    subprocess.run(["wpctl", "set-default", str(node_id)], check=False)


def retarget(stream: int, target: str | None) -> None:
    if target is None:
        subprocess.run(["pw-metadata", "-d", str(stream), "target.object"], capture_output=True, check=False)
    else:
        subprocess.run(["pw-metadata", str(stream), "target.object", target], capture_output=True, check=False)


def systemctl(*args: str) -> int:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True).returncode


def service_active() -> bool:
    return systemctl("is-active", "--quiet", SERVICE) == 0


# ---------------------------------------------------------------- install

UNIT = f"""[Unit]
Description=subvox head-tracked spatial audio (Powerbeats)
After=pipewire.service wireplumber.service filter-chain.service
Wants=filter-chain.service

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart=/usr/bin/python3 -u -m subvox spatial
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""

WRAPPER_SH = f"""#!/bin/sh
# subvox from anywhere; see {REPO}
cd "{REPO}" && exec python3 -m subvox "$@"
"""

WAYBAR_SNIPPET = """  "image#spatial": {
    "exec": "~/.config/athanor/waybar/spatial.py icon",
    "size": 22,
    "interval": 1,
    "on-click": "~/.local/bin/subvox ctl toggle",
    "on-click-middle": "~/.local/bin/subvox ctl recentre",
    "on-click-right": "~/.local/bin/subvox ctl settings",
    "on-scroll-up": "~/.local/bin/subvox ctl linger +1",
    "on-scroll-down": "~/.local/bin/subvox ctl linger -1"
  }"""


def install() -> None:
    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.write_text(UNIT)
    systemctl("daemon-reload")
    WRAPPER.parent.mkdir(parents=True, exist_ok=True)
    WRAPPER.write_text(WRAPPER_SH)
    WRAPPER.chmod(0o755)
    print(f"wrote {UNIT_PATH}\nwrote {WRAPPER}\nWaybar module (add to modules-right):\n{WAYBAR_SNIPPET}")


# ---------------------------------------------------------------- on / off

def ensure_sink(timeout: float = 8.0) -> int | None:
    g = graph()
    if g["spatial"] is None:
        subprocess.run([str(REPO / "tools/install-spatial.sh")], check=False, capture_output=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            g = graph()
            if g["spatial"] is not None:
                break
            time.sleep(0.3)
    return g["spatial"]


def on() -> None:
    if not UNIT_PATH.exists():
        install()
    s = Settings(); s.set("enabled", True); s.save()
    stray_daemon_stop()
    systemctl("enable", "--now", SERVICE)
    sink = ensure_sink()
    if sink is None:
        sys.exit("spatial sink did not appear; check `systemctl --user status filter-chain`")
    g = graph()
    set_default(sink)
    for st in g["streams"]:
        if g["buds"] in g["feeds"][st] or not g["feeds"][st]:
            retarget(st, SINK)
    print(f"spatial on: default sink {SINK}, daemon {'running' if service_active() else 'starting'}")


def off() -> None:
    s = Settings(); s.set("enabled", False); s.save()
    systemctl("disable", "--now", SERVICE)
    stray_daemon_stop()
    g = graph()
    if g["buds"] is not None:
        set_default(g["buds"])
    for st in g["streams"]:
        if g["spatial"] in g["feeds"][st]:
            retarget(st, None)          # follows the default again
    print(f"spatial off: default sink {g['buds_name'] or '(buds not connected)'}, plain {g['codec'] or 'A2DP'}")


def stray_daemon_stop() -> None:
    """A `subvox spatial` started by hand (not the service) would fight the service for AAP."""
    try:
        pid = int(PID_FILE.read_text())
        with open(f"/proc/{pid}/cgroup") as f:
            if SERVICE in f.read():
                return
        os.kill(pid, signal.SIGINT)
        time.sleep(0.8)
    except (OSError, ValueError):
        pass


def toggle() -> None:
    if Settings()["enabled"] and service_active():
        off()
    else:
        on()


# ---------------------------------------------------------------- status / settings

def status(as_json: bool) -> None:
    s = Settings()
    st = read_status()
    g = graph()
    d = {"settings": s.data, "daemon": st, "service": service_active(), "default_sink": g["default"],
         "buds_connected": g["buds"] is not None, "codec": g["codec"], "spatial_sink": g["spatial"] is not None,
         "streams": {g["stream_names"][k]: ("spatial" if g["spatial"] in v else "buds" if g["buds"] in v else "other")
                     for k, v in g["feeds"].items()}}
    if as_json:
        print(json.dumps(d))
        return
    on_ = s["enabled"] and d["service"]
    print(f"spatial: {'on' if on_ else 'off'}  (service {'active' if d['service'] else 'inactive'}, "
          f"default sink {g['default']}, buds {'connected, ' + str(g['codec']) if g['buds'] is not None else 'not connected'})")
    if st.get("stream") != "stopped":
        print(f"daemon: stream {st.get('stream')}  {st.get('hz', 0):.1f} Hz  yaw {st.get('yaw', 0):+.1f}° "
              f"pitch {st.get('pitch', 0):+.1f}° roll {st.get('roll', 0):+.1f}°  sent {st.get('sent', 0)}")
    print("settings: " + "  ".join(f"{k}={v}" for k, v in s.data.items() if k != "enabled"))
    for name, where in d["streams"].items():
        print(f"stream {name}: {where}")


def set_value(key: str, value: str) -> None:
    s = Settings()
    try:
        s.set(key, value)
    except KeyError:
        sys.exit(f"unknown setting {key}; one of {', '.join(k for k in s.data if k != 'enabled')}")
    s.save()
    print(f"{key} = {s[key]}")


def linger(arg: str) -> None:
    s = Settings()
    cur = s["linger"]
    new = cur + float(arg) if arg[0] in "+-" else float(arg)
    s.set("linger", new); s.save()
    print(f"linger = {s['linger']:.0f} s")


def recentre() -> None:
    try:
        os.kill(int(PID_FILE.read_text()), signal.SIGUSR1)
        print("recentred")
    except (OSError, ValueError):
        sys.exit("no running `subvox spatial`")


def settings_popup() -> None:
    """Open the GTK settings popup, or close it if it is already open."""
    try:
        pid = int(POPUP_PID.read_text())
        os.kill(pid, signal.SIGTERM)
        POPUP_PID.unlink(missing_ok=True)
        return
    except (OSError, ValueError):
        pass
    POPUP_PID.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["/usr/bin/python3", str(REPO / "tools/spatial-settings.py")],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    POPUP_PID.write_text(str(p.pid))


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "on": on()
    elif cmd == "off": off()
    elif cmd == "toggle": toggle()
    elif cmd == "status": status("--json" in rest)
    elif cmd == "set" and len(rest) == 2: set_value(*rest)
    elif cmd == "linger" and len(rest) == 1: linger(rest[0])
    elif cmd == "recentre": recentre()
    elif cmd == "settings": settings_popup()
    elif cmd == "install": install()
    else:
        sys.exit(__doc__)
