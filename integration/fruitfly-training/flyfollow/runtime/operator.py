"""Operator console: the safety keys, mode commands, settings and a live status panel (curses, macOS Terminal).

SAFETY FIRST: space = kill emergency (motors off, no confirm), l = kill land. Both publish immediately from
the key handler and do not depend on any other process being healthy. The console runs the terminal in raw
mode, so Ctrl+C is a key here (it lands, then quits) and never a signal to the rest of the system.

Keys: t takeoff (press twice), f FOLLOW, h HOLD, r RETURN, g GUIDE, k lock the best person track, u unlock,
/ command through the intent parser ("find bottle", "follow", "stop"), s edit a setting, c controller fly/pid,
q quit (press twice, lands first). While typing a command: Esc cancels, Ctrl+E = emergency, Ctrl+L = land.

    python -m flyfollow.runtime.operator                   # interactive (curses; --plain for a line UI)
    python -m flyfollow.runtime.operator --script cmds.txt # timed commands, non-interactive

Script format, one command per line, time in seconds from start ("#" comments):
    0.5 takeoff            1.0 wait tello_ack 5      2.0 follow        2.5 lock auto
    3.0 cmd find bottle    4.0 set follow_distance_m 2.5             5.0 controller pid
    6.0 mode HOLD          8.0 land                  (also: hold return guide unlock emergency lock <id>)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

from flyfollow.interfaces import IMG_W
from flyfollow.runtime.bus import Publisher, Subscriber
from flyfollow.runtime.messages import MODES, RC_OWNER, SETTINGS_DEFAULTS, msg

NAME = "operator"
TOPICS = ["tello_state", "tello_ack", "rc_sent", "rc", "ctrl_status", "mode", "mode_cmd", "target", "say", "cue",
          "find_status", "det", "lock", "settings", "kill", "health", "frame"]
STALE_S = {"tello_state": 1.0, "rc_sent": 0.6, "det": 1.0}
BATTERY_WARN = 25

# ------------------------------------------------------------------------------------------------ intent fallback
OBJECTS = {  # spoken word -> (detector class, assumed height m)
    "water bottle": ("bottle", 0.22), "bottle": ("bottle", 0.22), "cup": ("cup", 0.10), "mug": ("cup", 0.10),
    "phone": ("cell phone", 0.15), "cell phone": ("cell phone", 0.15), "remote": ("remote", 0.18),
    "book": ("book", 0.23), "backpack": ("backpack", 0.45), "bag": ("backpack", 0.45), "laptop": ("laptop", 0.25),
    "chair": ("chair", 0.9), "keys": ("keys", 0.08), "wallet": ("wallet", 0.10), "glasses": ("glasses", 0.05),
}
_FIND = re.compile(r"^(?:please\s+)?(?:find|where(?:'s| is| are)|look for|search for|get)\s+"
                   r"(?:my\s+|the\s+|a\s+)?(.+?)\s*[?.!]*$")


def builtin_parse(text: str) -> dict | None:
    """Minimal keyword intent rules (used when flyfollow.runtime.intent is missing or returns None)."""
    s = " ".join(text.lower().strip().split())
    if not s:
        return None
    m = _FIND.match(s)
    if m:
        obj = m.group(1)
        cls, h = OBJECTS.get(obj, OBJECTS.get(obj.split()[-1], (obj, None)))
        target = {"cls": cls, "prompt": obj}
        if h is not None:
            target["height_m"] = h
        return msg("mode_cmd", mode="FIND", source="operator", target=target)
    words = {
        "FOLLOW": ("follow", "follow me", "come", "resume"),
        "HOLD": ("stop", "stay", "hold", "wait", "pause", "hover"),
        "RETURN": ("return", "come back", "go back"),
        "GUIDE": ("guide", "guide me", "take me there"),
        "LAND": ("land", "land now"),
    }
    for mode, keys in words.items():
        if s in keys or s.rstrip(".!") in keys:
            return msg("mode_cmd", mode=mode, source="operator")
    up = s.upper()
    if up in MODES:
        return msg("mode_cmd", mode=up, source="operator")
    return None


def parse_intent(text: str) -> tuple[dict | None, str]:
    """(message, which parser) via R5's flyfollow.runtime.intent.parse, falling back to builtin_parse."""
    try:
        from flyfollow.runtime.intent import parse  # R5
    except Exception:
        parse = None
    if parse is not None:
        try:
            try:
                out = parse(text, source="operator")
            except TypeError:
                out = parse(text)
        except Exception:
            out = None
        if out:
            out = dict(out)
            out.setdefault("topic", "mode_cmd")
            out.setdefault("t", time.time())
            if out["topic"] == "mode_cmd":
                out["source"] = "operator"
            return out, "intent"
    return builtin_parse(text), "builtin"


def pick_user_track(det: dict | None, img_w: int = IMG_W) -> int | None:
    """The largest, most central person track in a det message (height x centrality)."""
    if not det:
        return None
    w = det.get("img_w") or img_w
    best, score = None, -1.0
    for d in det.get("dets", []):
        if d.get("cls") != "person" or d.get("track_id") is None:
            continue
        x1, y1, x2, y2 = d["bbox"]
        cx = 0.5 * (x1 + x2)
        s = max(0.0, y2 - y1) * (1.0 - 0.6 * min(1.0, abs(cx - w / 2) / (w / 2))) * (0.5 + 0.5 * float(d.get("conf", 1)))
        if s > score:
            best, score = int(d["track_id"]), s
    return best


def parse_value(key: str, text: str):
    """Setting value typed by the operator, cast like its default."""
    text = text.strip()
    default = SETTINGS_DEFAULTS.get(key)
    if text.lower() in ("none", "null", ""):
        return None
    if isinstance(default, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(text))
    if isinstance(default, float):
        return float(text)
    try:
        return json.loads(text)
    except ValueError:
        return text


def _f(v, spec: str = "{}") -> str:
    if v is None:
        return "-"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def _sticks(m: dict) -> str:
    return " ".join(f"{k} {_f(m.get(k)):>4}" for k in ("lr", "fb", "ud", "yaw"))


# ------------------------------------------------------------------------------------------------ console core
@dataclass
class Console:
    """UI-independent operator logic: actions publish, ingest() tracks status. Used by curses, plain and script modes."""

    pub: Publisher
    label: str = ""
    last: dict[str, dict] = field(default_factory=dict)
    seen: dict[str, float] = field(default_factory=dict)
    settings: dict = field(default_factory=lambda: dict(SETTINGS_DEFAULTS))
    modes: collections.deque = field(default_factory=lambda: collections.deque(maxlen=6))
    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=200))
    health: dict[str, tuple[float, str]] = field(default_factory=dict)
    lock_id: int | None = None
    n_cmd: int = 0
    n_notes: int = 0
    t0: float = field(default_factory=time.time)

    def note(self, text: str) -> None:
        self.events.append((time.time(), text))
        self.n_notes += 1

    def send(self, m: dict) -> dict:
        self.pub.publish(m)
        return m

    # ---- actions
    def kill(self, action: str) -> None:
        self.send(msg("kill", action=action, source="operator"))
        self.note(f"KILL {action.upper()} sent")

    def tello_cmd(self, cmd: str, **args) -> str:
        self.n_cmd += 1
        cid = f"op-{self.n_cmd}"
        m = msg("tello_cmd", id=cid, cmd=cmd)
        if args:
            m["args"] = args
        self.send(m)
        self.note(f"tello_cmd {cmd} ({cid}) sent")
        return cid

    def mode(self, mode: str, target: dict | None = None) -> None:
        mode = mode.upper()
        if mode not in MODES:
            self.note(f"unknown mode {mode}")
            return
        m = msg("mode_cmd", mode=mode, source="operator")
        if target:
            m["target"] = target
        self.send(m)
        self.note(f"mode_cmd {mode}" + (f" {target}" if target else ""))

    def lock(self, track_id: int | None) -> None:
        self.send(msg("lock", track_id=track_id, source="operator"))
        self.lock_id = track_id
        self.note("unlock sent" if track_id is None else f"lock track {track_id}")

    def lock_auto(self) -> int | None:
        tid = pick_user_track(self.last.get("det"))
        if tid is None:
            self.note("lock: no person track in the latest det")
            return None
        self.lock(tid)
        return tid

    def command(self, text: str) -> dict | None:
        text = text.strip()
        low = text.lower()
        if low in ("land", "emergency"):
            self.kill(low)
            return None
        if low == "takeoff":
            self.tello_cmd("takeoff")
            return None
        if low.startswith("lock"):
            arg = low[4:].strip()
            if arg in ("", "auto"):
                self.lock_auto()
            else:
                self.lock(None if arg == "none" else int(arg))
            return None
        if low.startswith("set "):
            parts = text.split(None, 2)
            if len(parts) == 3:
                self.set_setting(parts[1], parts[2])
            return None
        m, how = parse_intent(text)
        if m is None:
            self.note(f"not understood: {text!r}")
            return None
        self.send(m)
        self.note(f"{how}: {text!r} -> {m.get('topic')} {m.get('mode', '')} {m.get('target', '') or ''}".rstrip())
        return m

    def set_setting(self, key: str, value_text: str) -> None:
        key = key.strip()
        if key not in SETTINGS_DEFAULTS:
            self.note(f"unknown setting {key!r} (keys: {', '.join(SETTINGS_DEFAULTS)})")
            return
        try:
            v = parse_value(key, value_text)
        except ValueError:
            self.note(f"bad value for {key}: {value_text!r}")
            return
        self.settings[key] = v
        self.send(msg("settings", **{key: v}, source="operator"))
        self.note(f"settings {key} = {v!r}")

    def toggle_controller(self) -> str:
        cur = (self.last.get("ctrl_status") or {}).get("controller") or self.settings.get("controller", "fly")
        new = "pid" if str(cur).lower().startswith("fly") else "fly"
        self.settings["controller"] = new
        self.send(msg("settings", controller=new, source="operator"))
        self.note(f"controller -> {new}")
        return new

    # ---- status
    def ingest(self, m: dict) -> None:
        tp = m.get("topic", "?")
        now = time.time()
        self.seen[tp] = now
        self.last[tp] = m
        if tp == "mode":
            self.modes.append((m.get("t", now), m.get("from"), m.get("to"), m.get("reason", "")))
        elif tp == "tello_ack":
            self.note(f"ack {m.get('cmd')} ({m.get('id')}): {'ok' if m.get('ok') else 'FAILED'} {m.get('detail', '')}")
        elif tp == "settings" and m.get("src_node") != self.pub.name:
            self.settings.update({k: v for k, v in m.items() if k in SETTINGS_DEFAULTS})
        elif tp == "lock" and m.get("src_node") != self.pub.name:
            self.lock_id = m.get("track_id")
        elif tp == "health":
            key = m.get("key") or m.get("text", "")
            if m.get("ok"):
                self.health.pop(key, None)
            else:
                self.health[key] = (now, m.get("text", key))
        elif tp == "kill" and m.get("src_node") != self.pub.name:
            self.note(f"kill {m.get('action')} from {m.get('src_node')}")
        elif tp == "say":
            self.note(f"say[{m.get('priority')}]: {m.get('text')}")

    def age(self, tp: str) -> float | None:
        t = self.seen.get(tp)
        return None if t is None else time.time() - t

    def current_mode(self) -> str:
        if self.modes:
            return str(self.modes[-1][2])
        return str((self.last.get("ctrl_status") or {}).get("mode") or "?")

    def warnings(self) -> list[str]:
        now = time.time()
        out = []
        for tp, lim in STALE_S.items():
            a = self.age(tp)
            if a is None:
                if now - self.t0 > 5.0:
                    out.append(f"no {tp} yet")
            elif a > lim:
                out.append(f"{tp} stale {a:.1f}s")
        st = self.last.get("tello_state") or {}
        if isinstance(st.get("bat_pct"), (int, float)) and st["bat_pct"] < BATTERY_WARN:
            out.append(f"BATTERY {st['bat_pct']}%")
        if isinstance(st.get("video_age_s"), (int, float)) and st["video_age_s"] > 1.0:
            out.append(f"video age {st['video_age_s']:.1f}s")
        if st and st.get("video_ok") is False:
            out.append("video not ok")
        mode = self.current_mode()
        if RC_OWNER.get(mode) and (self.age("rc") or 99) > 1.0:
            out.append(f"no rc from {RC_OWNER[mode]} in {mode}")
        gov = (self.last.get("rc") or {}).get("gov") or {}
        if gov.get("safety") or gov.get("clamped"):
            out.append("gov: " + ",".join(gov.get("reasons") or ["safety" if gov.get("safety") else "clamped"]))
        out.extend(text for _, (_, text) in sorted(self.health.items()))
        return out

    def status_lines(self) -> list[tuple[str, str]]:
        """(text, style) lines; style in "", "title", "mode", "warn", "dim"."""
        now = time.time()
        L = self.last
        st, rs, rc, cs = (L.get(k) or {} for k in ("tello_state", "rc_sent", "rc", "ctrl_status"))
        out = [(f"FLYFOLLOW OPERATOR  {self.label}  {time.strftime('%H:%M:%S')}", "title")]
        since = f"{now - self.modes[-1][0]:.0f}s" if self.modes else "-"
        ctrl = cs.get("controller") or self.settings.get("controller")
        out.append((f"MODE   {self.current_mode():<11} since {since:<6} controller {ctrl}   "
                    f"brain {_f(rc.get('brain_tick_ms'), '{:.1f}')} ms   lock {_f(self.lock_id)}", "mode"))
        out.append((f"DRONE  bat {_f(st.get('bat_pct'))}%  h {_f(st.get('h_cm'))} cm  tof {_f(st.get('tof_cm'))} cm  "
                    f"flying {_f(st.get('flying'))}  video age {_f(st.get('video_age_s'), '{:.2f}')} s  "
                    f"temp {_f(st.get('temph_c'))} C  sending {_f(st.get('sending'))}  "
                    f"(state {_f(self.age('tello_state'), '{:.1f}')} s ago)", ""))
        dry = rs.get("dry_run")
        live = "DRY RUN" if dry else ("LIVE" if dry is False else "-")
        out.append((f"RC OUT {_sticks(rs)}  src {_f(rs.get('src'))}  {live}", "warn" if dry is False else ""))
        out.append((f"RC REQ {_sticks(rc)}  src {_f(rc.get('src'))}  mode {_f(rc.get('mode'))}", "dim"))
        out.append((f"TARGET valid {_f(cs.get('target_valid'))}  range {_f(cs.get('range_m'), '{:.2f}')} m  "
                    f"bearing {_f(cs.get('bearing_deg'), '{:+.1f}')} deg  in band {_f(cs.get('in_band'))}", ""))
        det = L.get("det") or {}
        persons = [d for d in det.get("dets", []) if d.get("cls") == "person"]
        tracks = [d.get("track_id") for d in persons][:6]
        out.append((f"DET    {len(det.get('dets', []))} boxes, {len(persons)} persons, tracks {tracks}  "
                    f"src {_f(det.get('src'))}  age {_f(self.age('det'), '{:.1f}')} s", "dim"))
        fs = L.get("find_status")
        if fs:
            out.append((f"FIND   {fs.get('state')}  hops {fs.get('hops')}  {_f(fs.get('elapsed_s'), '{:.0f}')} s  "
                        f"searched {fs.get('searched')}", ""))
        say = L.get("say")
        out.append((f"SAY    {say.get('text')!r} ({now - say.get('t', now):.0f}s ago)" if say else "SAY    -", ""))
        cue = L.get("cue")
        out.append((f"CUE    {cue.get('kind')} {_f(cue.get('strength'), '{:.2f}')}" if cue else "CUE    -", ""))
        trans = " | ".join(f"{time.strftime('%H:%M:%S', time.localtime(t))} {a}->{b}" for t, a, b, _ in list(self.modes)[-4:])
        out.append((f"MODES  {trans or '-'}", "dim"))
        if self.modes and self.modes[-1][3]:
            out.append((f"       reason: {self.modes[-1][3]}", "dim"))
        warns = self.warnings()
        out.append(("WARN   " + (" | ".join(warns) if warns else "none"), "warn" if warns else "dim"))
        return out


KEYS_HELP = ("[space] EMERGENCY  [l] LAND  [t]x2 takeoff  [f] follow [h] hold [r] return [g] guide  [k] lock [u] unlock  "
             "[/] command [s] setting [c] fly/pid  [q]x2 quit")


# ------------------------------------------------------------------------------------------------ key handling
@dataclass
class KeyState:
    armed: dict[str, float] = field(default_factory=dict)  # double-press keys -> time of first press
    input_kind: str | None = None  # "cmd" | "set" while typing
    buf: str = ""
    quit: bool = False


def handle_key(c: Console, ks: KeyState, ch: int) -> None:
    """One key press. Kill keys act immediately in every state."""
    now = time.time()
    if ch in (5,):  # Ctrl+E
        c.kill("emergency")
        return
    if ch in (12,):  # Ctrl+L
        c.kill("land")
        return
    if ch == 3:  # Ctrl+C: land, then quit
        c.kill("land")
        ks.quit = True
        return
    if ks.input_kind:
        if ch in (10, 13):
            text, kind = ks.buf, ks.input_kind
            ks.input_kind, ks.buf = None, ""
            if kind == "cmd":
                c.command(text)
            else:
                parts = text.replace("=", " ").split(None, 1)
                if len(parts) == 2:
                    c.set_setting(parts[0], parts[1])
                elif parts:
                    c.note(f"usage: <key> <value>  (current {parts[0]} = {c.settings.get(parts[0])!r})")
        elif ch == 27:
            ks.input_kind, ks.buf = None, ""
        elif ch in (8, 127, 263):
            ks.buf = ks.buf[:-1]
        elif 32 <= ch < 127:
            ks.buf += chr(ch)
        return
    if ch == ord(" "):
        c.kill("emergency")
        return
    k = chr(ch) if 0 <= ch < 256 else ""
    if k == "l":
        c.kill("land")
    elif k == "t":
        if now - ks.armed.pop("t", 0.0) < 2.0:
            c.tello_cmd("takeoff")
        else:
            ks.armed["t"] = now
            c.note("press t again within 2 s to take off")
    elif k == "q":
        if now - ks.armed.pop("q", 0.0) < 2.0:
            c.kill("land")
            ks.quit = True
        else:
            ks.armed["q"] = now
            c.note("press q again within 2 s to quit (sends land first)")
    elif k in ("f", "h", "r", "g"):
        c.mode({"f": "FOLLOW", "h": "HOLD", "r": "RETURN", "g": "GUIDE"}[k])
    elif k == "k":
        c.lock_auto()
    elif k == "u":
        c.lock(None)
    elif k == "c":
        c.toggle_controller()
    elif k == "/":
        ks.input_kind, ks.buf = "cmd", ""
    elif k == "s":
        ks.input_kind, ks.buf = "set", ""
        c.note("settings: " + ", ".join(f"{a}={b}" for a, b in c.settings.items()))


# ------------------------------------------------------------------------------------------------ curses UI
def run_curses(c: Console, sub: Subscriber) -> None:
    os.environ.setdefault("ESCDELAY", "25")
    import curses

    def ui(scr) -> None:
        curses.raw()  # Ctrl+C / Ctrl+Z become keys
        curses.noecho()
        scr.keypad(True)
        scr.timeout(40)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        styles = {"": curses.A_NORMAL, "title": curses.A_BOLD | curses.A_REVERSE, "mode": curses.A_BOLD,
                  "warn": curses.A_BOLD, "dim": curses.A_DIM}
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                bg = -1
            except curses.error:
                bg = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_RED, bg)
            curses.init_pair(2, curses.COLOR_CYAN, bg)
            styles["warn"] = curses.color_pair(1) | curses.A_BOLD
            styles["mode"] = curses.color_pair(2) | curses.A_BOLD
        ks = KeyState()
        last_draw = 0.0
        while not ks.quit:
            ch = scr.getch()
            while ch != -1:  # handle every queued key before anything else
                if ch != curses.KEY_RESIZE:
                    handle_key(c, ks, ch)
                if ks.quit:
                    return
                ch = scr.getch() if not ks.quit else -1
                scr.timeout(0)
            scr.timeout(40)
            for m in sub.drain(500):
                c.ingest(m)
            if time.monotonic() - last_draw < 0.1:
                continue
            last_draw = time.monotonic()
            h, w = scr.getmaxyx()
            scr.erase()

            def put(y: int, text: str, attr=curses.A_NORMAL, h=h, w=w) -> None:
                if 0 <= y < h:
                    try:
                        scr.addnstr(y, 0, text, max(0, w - 1), attr)
                    except curses.error:
                        pass

            y = 0
            for text, style in c.status_lines():
                put(y, text, styles.get(style, curses.A_NORMAL))
                y += 1
            y += 1
            n_log = max(0, h - y - 3)
            put(y, "---- events ----", curses.A_DIM)
            for t, text in list(c.events)[-n_log:] if n_log else []:
                y += 1
                put(y, f"{time.strftime('%H:%M:%S', time.localtime(t))} {text}")
            if ks.input_kind:
                prompt = "command> " if ks.input_kind == "cmd" else "setting (key value)> "
                put(h - 2, prompt + ks.buf + "_   (Enter send, Esc cancel, Ctrl+E EMERGENCY, Ctrl+L land)", curses.A_BOLD)
            put(h - 1, KEYS_HELP, curses.A_REVERSE)
            scr.refresh()

    curses.wrapper(ui)


# ------------------------------------------------------------------------------------------------ plain UI (Windows / no curses)
def run_plain(c: Console, sub: Subscriber) -> None:
    """Line UI: status every second, keys without Enter. Commands and settings are read with input()."""
    print(KEYS_HELP, flush=True)
    getkey, restore = _raw_keys()
    ks = KeyState()
    last = 0.0
    printed = 0
    try:
        while not ks.quit:
            ch = getkey(0.05)
            if ch is not None:
                if ch in (ord("/"), ord("s")):
                    restore()
                    kind = "cmd" if ch == ord("/") else "set"
                    try:
                        text = input("command> " if kind == "cmd" else "setting (key value)> ")
                    except EOFError:
                        text = ""
                    getkey, restore = _raw_keys()
                    ks.input_kind, ks.buf = kind, text
                    handle_key(c, ks, 13)
                else:
                    handle_key(c, ks, ch)
            for m in sub.drain(500):
                c.ingest(m)
            new = min(c.n_notes - printed, len(c.events))
            for t, text in list(c.events)[len(c.events) - new:]:
                print(f"{time.strftime('%H:%M:%S', time.localtime(t))} {text}\r", flush=True)
            printed = c.n_notes
            if time.monotonic() - last > 1.0:
                last = time.monotonic()
                lines = c.status_lines()
                print(" || ".join(t for t, _ in lines[1:4]) + " || " + lines[-1][0] + "\r", flush=True)
    finally:
        restore()


def _raw_keys():
    """(getkey(timeout_s) -> int | None, restore()) for the current terminal."""
    if os.name == "nt":
        import msvcrt

        def getkey(timeout_s: float):
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                if msvcrt.kbhit():
                    return ord(msvcrt.getwch())
                time.sleep(0.01)
            return None

        return getkey, lambda: None
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)

    def getkey(timeout_s: float):
        r, _, _ = select.select([fd], [], [], timeout_s)
        return os.read(fd, 1)[0] if r else None

    return getkey, lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ------------------------------------------------------------------------------------------------ script mode
def parse_script(text: str) -> list[tuple[float, str]]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        t, _, cmd = line.partition(" ")
        out.append((float(t), cmd.strip()))
    return sorted(out, key=lambda x: x[0])


def run_script(c: Console, sub: Subscriber, steps: list[tuple[float, str]], linger_s: float = 1.0,
               echo: bool = True) -> int:
    """Run timed commands; returns the number of failed waits (0 = ok)."""
    failures = 0
    t0 = time.time()
    shown = 0

    def pump(until: float) -> None:
        nonlocal shown
        while True:
            m = sub.recv(max(0.0, min(0.05, until - time.time())))
            if m is not None:
                c.ingest(m)
                if echo and m.get("topic") in ("mode", "say", "tello_ack", "cue", "kill", "health"):
                    body = {k: v for k, v in m.items() if k not in ("topic", "t", "src_node")}
                    print(f"[operator] <- {m['topic']} {json.dumps(body)}", flush=True)
            if time.time() >= until:
                break

    for t_at, cmd in steps:
        pump(t0 + t_at)
        if echo:
            print(f"[operator] {time.time() - t0:6.2f}s -> {cmd}", flush=True)
        words = cmd.split()
        op, args = (words[0].lower(), words[1:]) if words else ("", [])
        if op == "wait":
            topic = args[0]
            deadline = time.time() + (float(args[1]) if len(args) > 1 else 5.0)
            since = time.time()
            while time.time() < deadline and c.seen.get(topic, 0.0) < since:
                pump(min(deadline, time.time() + 0.05))
            if c.seen.get(topic, 0.0) < since:
                failures += 1
                print(f"[operator] wait {topic}: TIMEOUT", flush=True)
        elif op in ("takeoff",):
            c.tello_cmd("takeoff")
        elif op in ("land", "emergency"):
            c.kill(op)
        elif op in ("follow", "hold", "return", "guide"):
            c.mode(op.upper())
        elif op == "mode":
            target = {"cls": args[1], "prompt": " ".join(args[1:])} if len(args) > 1 else None
            c.mode(args[0], target)
        elif op == "lock":
            if not args or args[0] == "auto":
                c.lock_auto()
            else:
                c.lock(int(args[0]))
        elif op == "unlock":
            c.lock(None)
        elif op in ("cmd", "say"):
            c.command(" ".join(args))
        elif op == "set" and len(args) >= 2:
            c.set_setting(args[0], " ".join(args[1:]))
        elif op == "controller":
            if args:
                c.set_setting("controller", args[0])
            else:
                c.toggle_controller()
        elif op == "tello_cmd" and args:
            c.tello_cmd(args[0], **(json.loads(" ".join(args[1:])) if len(args) > 1 else {}))
        else:
            print(f"[operator] unknown script command: {cmd!r}", flush=True)
            failures += 1
    pump(time.time() + linger_s)
    return failures


# ------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="operator console (kill keys, modes, settings, status)")
    ap.add_argument("--script", default=None, help="run timed commands from a file instead of the interactive UI")
    ap.add_argument("--linger", type=float, default=1.0, help="script mode: seconds to keep listening at the end")
    ap.add_argument("--plain", action="store_true", help="line UI instead of curses")
    ap.add_argument("--label", default=None, help="shown in the title (default: $FLYFOLLOW_MODE)")
    a = ap.parse_args(argv)
    sub = Subscriber(TOPICS)
    pub = Publisher(NAME)
    label = a.label or os.environ.get("FLYFOLLOW_MODE", "")
    if os.environ.get("FLYFOLLOW_SESSION"):
        label += f"  session {os.environ['FLYFOLLOW_SESSION']}"
    c = Console(pub, label=label.strip())
    code = 0
    try:
        if a.script:
            with open(a.script) as f:
                steps = parse_script(f.read())
            code = 1 if run_script(c, sub, steps, linger_s=a.linger) else 0
        elif a.plain or not sys.stdout.isatty():
            run_plain(c, sub)
        else:
            try:
                run_curses(c, sub)
            except ImportError:
                run_plain(c, sub)
    except KeyboardInterrupt:
        c.kill("land")
    finally:
        time.sleep(0.05)
        pub.close()
        sub.close()
    sys.exit(code)


if __name__ == "__main__":
    main()

