#!/usr/bin/python3
"""Settings popup for `subvox spatial`, opened from the Waybar module (`subvox ctl settings`).

GTK 3 on a layer-shell surface, so it drops down under the bar like a menu instead of tiling.
Every change is written to ~/.config/subvox/spatial.json at once; the daemon reloads it within
half a second. The on/off switch runs `subvox ctl on|off`, which also moves the audio routing.
Escape or clicking outside closes it. Needs python3-gi, gir1.2-gtk-3.0, gir1.2-gtklayershell-0.1
(all system packages here); the subvox package itself stays stdlib-only.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gdk, GLib, Gtk, GtkLayerShell  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from subvox.settings import LIMITS, Settings, read_status  # noqa: E402

PID_FILE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "subvox/settings-popup.pid"
BAR_CONFIG = Path.home() / ".config/athanor/waybar/config.jsonc"


def bar_edge() -> tuple[str, int]:
    """Which screen edge the Athanor bar is on and how tall it is, so the popup hangs off it."""
    try:
        text = BAR_CONFIG.read_text()
        pos = re.search(r'"position"\s*:\s*"(\w+)"', text)
        height = re.search(r'"height"\s*:\s*(\d+)', text)
        return (pos.group(1) if pos else "top"), (int(height.group(1)) if height else 30)
    except OSError:
        return "top", 30
CSS = b"""
window { background: alpha(@theme_bg_color, 0.96); border-radius: 10px; }
.title { font-weight: bold; font-size: 1.1em; }
.hint  { opacity: 0.7; font-size: 0.9em; }
"""

FIELDS = [
    # key, label, hint, step, digits
    ("linger", "Linger", "seconds before the stage drifts back to where you face (0 = never)", 1.0, 0),
    ("lead", "Lead", "seconds of head motion to predict; covers the Bluetooth delay", 0.01, 2),
    ("deadband", "Deadband", "degrees a speaker must move before the HRTF is rebuilt", 0.25, 2),
    ("spread", "Speakers", "half-angle of the virtual speaker pair, degrees", 5.0, 0),
]


def ctl(*args: str) -> None:
    subprocess.Popen(["/usr/bin/python3", "-m", "subvox", "ctl", *args], cwd=REPO,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def service_active() -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", "subvox-spatial"]).returncode == 0


class Popup(Gtk.Window):
    def __init__(self) -> None:
        super().__init__(title="Stage settings")
        self.settings = Settings()
        self.set_default_size(340, -1)
        self.set_resizable(False)
        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.ON_DEMAND)
        edge, height = bar_edge()
        vertical = GtkLayerShell.Edge.BOTTOM if edge == "bottom" else GtkLayerShell.Edge.TOP
        GtkLayerShell.set_anchor(self, vertical, True)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.RIGHT, True)
        GtkLayerShell.set_margin(self, vertical, height + 10)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.RIGHT, 12)
        GtkLayerShell.set_namespace(self, "subvox-settings")

        prov = Gtk.CssProvider(); prov.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8, margin=16)
        self.add(grid)
        row = 0
        title = Gtk.Label(label="Stage", xalign=0); title.get_style_context().add_class("title")
        grid.attach(title, 0, row, 2, 1)
        self.power = Gtk.Switch(halign=Gtk.Align.END)
        self.power.set_active(self.settings["enabled"] and service_active())
        self.power.connect("notify::active", self.on_power)
        grid.attach(self.power, 2, row, 1, 1); row += 1
        sub = Gtk.Label(label="off = plain stereo straight to the buds", xalign=0); sub.get_style_context().add_class("hint")
        grid.attach(sub, 0, row, 3, 1); row += 1
        self.status = Gtk.Label(label="", xalign=0, wrap=True, max_width_chars=42)
        self.status.get_style_context().add_class("hint")
        grid.attach(self.status, 0, row, 3, 1); row += 1
        grid.attach(Gtk.Separator(), 0, row, 3, 1); row += 1

        self.spins: dict[str, Gtk.SpinButton] = {}
        for key, label, hint, step, digits in FIELDS:
            lo, hi = LIMITS[key]
            adj = Gtk.Adjustment(value=self.settings[key], lower=lo, upper=hi, step_increment=step, page_increment=step * 5)
            spin = Gtk.SpinButton(adjustment=adj, digits=digits, numeric=True, halign=Gtk.Align.END)
            spin.set_width_chars(6)
            spin.connect("value-changed", self.on_spin, key)
            lab = Gtk.Label(label=label, xalign=0)
            grid.attach(lab, 0, row, 1, 1); grid.attach(spin, 2, row, 1, 1); row += 1
            h = Gtk.Label(label=hint, xalign=0, wrap=True, max_width_chars=42); h.get_style_context().add_class("hint")
            grid.attach(h, 0, row, 3, 1); row += 1
            self.spins[key] = spin

        self.elev = Gtk.Switch(halign=Gtk.Align.END, active=self.settings["elevation"])
        self.elev.connect("notify::active", self.on_elev)
        grid.attach(Gtk.Label(label="Elevation", xalign=0), 0, row, 1, 1); grid.attach(self.elev, 2, row, 1, 1); row += 1
        h = Gtk.Label(label="follow nods and tilts as well as turns", xalign=0); h.get_style_context().add_class("hint")
        grid.attach(h, 0, row, 3, 1); row += 1

        box = Gtk.Box(spacing=8, margin_top=8)
        rec = Gtk.Button(label="This way is front"); rec.connect("clicked", lambda *_: ctl("recentre"))
        cal = Gtk.Button(label="Calibrate…"); cal.connect("clicked", self.on_calibrate)
        close = Gtk.Button(label="Close"); close.connect("clicked", lambda *_: self.quit())
        box.pack_start(rec, False, False, 0); box.pack_start(cal, False, False, 0); box.pack_end(close, False, False, 0)
        grid.attach(box, 0, row, 3, 1)

        self.connect("key-press-event", self.on_key)
        self.connect("destroy", lambda *_: self.quit())
        GLib.timeout_add(1000, self.refresh)

    # -- handlers
    def on_power(self, sw: Gtk.Switch, _p) -> None:
        ctl("on" if sw.get_active() else "off")

    def on_spin(self, spin: Gtk.SpinButton, key: str) -> None:
        self.settings.set(key, spin.get_value()); self.settings.save()

    def on_elev(self, sw: Gtk.Switch, _p) -> None:
        self.settings.set("elevation", sw.get_active()); self.settings.save()

    def on_calibrate(self, *_) -> None:
        subprocess.Popen(["foot", "-e", "sh", "-c", f"cd {REPO} && python3 -m subvox ctl off; python3 -m subvox calibrate; "
                          f"python3 -m subvox ctl on; echo; read -p 'done, Enter closes'"])
        self.quit()

    def on_key(self, _w, ev) -> bool:
        if ev.keyval == Gdk.KEY_Escape:
            self.quit(); return True
        return False

    def refresh(self) -> bool:
        st = read_status()
        if st.get("stream") == "open":
            self.status.set_text(f"tracking {st.get('hz', 0):.0f} Hz · yaw {st.get('yaw', 0):+.0f}° "
                                 f"pitch {st.get('pitch', 0):+.0f}° roll {st.get('roll', 0):+.0f}° · "
                                 f"{'still' if st.get('still') else 'moving'} · {st.get('sent', 0)} updates")
        elif st.get("stream") == "stopped":
            self.status.set_text("daemon not running" if self.settings["enabled"] else "off")
        else:
            self.status.set_text("daemon up, waiting for the buds' motion stream")
        # keep the power switch honest if it was toggled from the bar meanwhile
        if self.settings.reload():
            for key, spin in self.spins.items():
                if abs(spin.get_value() - self.settings[key]) > 1e-9:
                    spin.set_value(self.settings[key])
        want = self.settings["enabled"] and service_active()
        if self.power.get_active() != want:
            self.power.handler_block_by_func(self.on_power); self.power.set_active(want); self.power.handler_unblock_by_func(self.on_power)
        return True

    def quit(self) -> None:
        PID_FILE.unlink(missing_ok=True)
        Gtk.main_quit()


def main() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    # a plain Python signal handler never runs while Gtk.main() blocks in C; GLib's does
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, lambda *_: (Gtk.main_quit(), False)[1])
    w = Popup()
    w.show_all()
    Gtk.main()
    PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
