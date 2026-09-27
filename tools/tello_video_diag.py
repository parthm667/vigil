#!/usr/bin/env python3
"""
Tello video diagnosis: find WHERE the camera chain breaks on this laptop (for "the phone app shows video,
our code does not", e.g. on a Mac).

Each layer is tested on its own, in order:
  1. packages    cv2 / av / djitellopy versions and their FFmpeg builds (cv2 and av each bundle one)
  2. ports       is another program holding UDP 8889 / 8890 / 11111 (lsof + a bind test)
  3. network     laptop IP towards the drone, route, macOS firewall state, which app launched Python
  4. link        'command' -> 'ok'
  5. state       state packets on UDP 8890 (does inbound UDP reach Python at all?)
  6. raw video   'streamon', then raw packets on UDP 11111: rate, sender, SPS / PPS / IDR timing. No decoder.
  7. decoders    each in its own process (a crash or hang cannot stop the run), on the same live stream:
                   pyav              PyAV, like flyfollow's VideoReader
                   cv2               OpenCV exactly like reachglass's VideoStream (5 s open timeout)
                   cv2_long_timeout  the same with a 30 s open timeout
                   cv2_after_av      OpenCV after importing PyAV first (djitellopy does that in the real app)
                   reachglass        reachglass.sources.TelloVideoSource itself
It ends with a verdict and saves everything to results/tello_video_diag_<stamp>.json.

Run with the app's venv, drone on the ground, laptop on TELLO-xxxxxx, the phone app closed, no other Tello
program running:
    python tools/tello_video_diag.py
    python tools/tello_video_diag.py --seconds 10      # watch each decoder longer
About 1.5 minutes. Nothing moves: it only sends command, battery?, streamon and streamoff.
"""

import argparse
import errno
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111
VIDEO_URL = f"udp://@0.0.0.0:{VIDEO_PORT}"
REACHGLASS_CV2_OPTIONS = "fflags;nobuffer|flags;low_delay|framedrop;1"  # reachglass/sources/video_stream.py
FLYFOLLOW_AV_OPTIONS = {"fflags": "nobuffer", "flags": "low_delay", "fifo_size": "5000000", "overrun_nonfatal": "1",
                        "probesize": "500000", "analyzeduration": "500000"}  # flyfollow/runtime/tello_io.py
DECODERS = ["pyav", "cv2", "cv2_long_timeout", "cv2_after_av", "reachglass"]
OPENCV_PACKAGES = ["opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"]


# ------------------------------------------------------------------ helpers
def say(status, stage, detail):
    print(f"[{status:^4}] {stage:<17} {detail}", flush=True)


def err_text(e):
    name = ""
    if e.errno is not None:
        name = errno.errorcode.get(e.errno, "")
    return f"{name} {e}".strip()


def run_tool(args, timeout=5):
    """Output of a system tool, or None if it is not there."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (p.stdout + p.stderr).strip()


def open_udp(port):
    """A UDP socket bound to port (0 = any free port), or the bind error as text."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("", port))
    except OSError as e:
        s.close()
        return None, err_text(e)
    return s, None


def send(sock, ip, text, timeout=3.0):
    """Send one SDK command, wait for the reply. Returns (reply or None, seconds, error or None)."""
    sock.setblocking(False)
    while True:
        try:
            sock.recvfrom(2048)  # drop stale replies
        except OSError:
            break
    t0 = time.time()
    try:
        sock.sendto(text.encode(), (ip, CONTROL_PORT))
    except OSError as e:
        return None, 0.0, err_text(e)
    sock.settimeout(timeout)
    try:
        data, addr = sock.recvfrom(2048)
    except socket.timeout:
        return None, time.time() - t0, "timeout"
    except OSError as e:
        return None, time.time() - t0, err_text(e)
    return data.decode(errors="ignore").strip(), time.time() - t0, None


def nal_types(buf):
    """H.264 NAL unit types after each 00 00 01 start code in buf."""
    types = []
    i = buf.find(b"\x00\x00\x01")
    while i != -1 and i + 3 < len(buf):
        types.append(buf[i + 3] & 0x1F)
        i = buf.find(b"\x00\x00\x01", i + 3)
    return types


# ------------------------------------------------------------------ layers 2 to 6 (this process)
def check_ports():
    result = {"lsof": None, "bind": {}}
    if sys.platform != "win32":
        result["lsof"] = run_tool(["lsof", "-nP", f"-iUDP:{CONTROL_PORT}", f"-iUDP:{STATE_PORT}", f"-iUDP:{VIDEO_PORT}"])
    for port in [CONTROL_PORT, STATE_PORT, VIDEO_PORT]:
        s, error = open_udp(port)
        if s is None:
            result["bind"][str(port)] = error
        else:
            result["bind"][str(port)] = "free"
            s.close()
    return result


def check_network(ip):
    result = {"local_ip": None, "route": None, "firewall": None}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ip, CONTROL_PORT))  # only picks the interface, sends nothing
        result["local_ip"] = s.getsockname()[0]
    except OSError as e:
        result["local_ip_error"] = err_text(e)
    finally:
        s.close()
    if sys.platform == "darwin":
        route = run_tool(["route", "-n", "get", ip])
        if route is not None:
            keep = []
            for line in route.splitlines():
                if "interface" in line or "gateway" in line:
                    keep.append(line.strip())
            result["route"] = keep
        fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
        lines = []
        for flag in ["--getglobalstate", "--getblockall", "--getstealthmode"]:
            lines.append(run_tool([fw, flag]))
        lines.append(run_tool([fw, "--getappblocked", os.path.realpath(sys.executable)]))
        result["firewall"] = lines
    return result


def watch_state(sock, seconds):
    result = {"packets": 0, "battery": None}
    t_end = time.time() + seconds
    while time.time() < t_end:
        sock.settimeout(max(0.01, t_end - time.time()))
        try:
            data, addr = sock.recvfrom(2048)
        except OSError:
            break
        result["packets"] += 1
        for field in data.decode(errors="ignore").split(";"):
            if field.startswith("bat:"):
                result["battery"] = field[4:]
    return result


def watch_video(sock, seconds, t0):
    """Raw packets on UDP 11111 until t0 + seconds. Times are seconds after t0 (the streamon)."""
    result = {"packets": 0, "bytes": 0, "senders": [], "first_packet_s": None, "nal_counts": {},
              "sps_s": [], "idr_s": []}
    senders = set()
    tail = b""
    t_end = t0 + seconds
    while time.time() < t_end:
        sock.settimeout(max(0.01, t_end - time.time()))
        try:
            data, addr = sock.recvfrom(65536)
        except OSError:
            break
        t = round(time.time() - t0, 2)
        if result["first_packet_s"] is None:
            result["first_packet_s"] = t
        result["packets"] += 1
        result["bytes"] += len(data)
        senders.add(f"{addr[0]}:{addr[1]}")
        for nal_type in nal_types(tail + data):  # a start code can straddle two packets
            name = {1: "slice", 5: "IDR", 7: "SPS", 8: "PPS"}.get(nal_type, str(nal_type))
            result["nal_counts"][name] = result["nal_counts"].get(name, 0) + 1
            if nal_type == 7:
                result["sps_s"].append(t)
            if nal_type == 5:
                result["idr_s"].append(t)
        tail = data[-3:]
    result["senders"] = sorted(senders)
    result["kbit_s"] = round(result["bytes"] * 8 / 1000 / seconds)
    return result


def drain(sock, seconds):
    """Packets already arriving before streamon (the stream was left on by an earlier program)."""
    count = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        sock.settimeout(max(0.01, t_end - time.time()))
        try:
            sock.recvfrom(65536)
        except OSError:
            break
        count += 1
    return count


# ------------------------------------------------------------------ layer 1 and 7 (child processes)
def child_info(seconds):
    from importlib import metadata

    result = {"packages": {}}
    for name in OPENCV_PACKAGES + ["av", "djitellopy", "numpy"]:
        try:
            result["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import av

        result["av_libavformat"] = list(av.library_versions.get("libavformat", ()))
    except ImportError as e:
        result["av_error"] = str(e)
    try:
        import cv2

        result["cv2"] = cv2.__version__
        lines = []
        for line in cv2.getBuildInformation().splitlines():
            if "FFMPEG" in line or "avformat" in line:
                lines.append(line.strip())
        result["cv2_ffmpeg"] = lines
    except ImportError as e:
        result["cv2_error"] = str(e)
    return result


def child_pyav(seconds):
    import av

    result = {"opened": False, "frames": 0}
    t0 = time.time()
    try:
        container = av.open(VIDEO_URL, format="h264", options=FLYFOLLOW_AV_OPTIONS, timeout=(10.0, 3.0))
    except Exception as e:
        result["open_s"] = round(time.time() - t0, 2)
        result["error"] = f"open: {type(e).__name__}: {e}"
        return result
    result["opened"] = True
    result["open_s"] = round(time.time() - t0, 2)
    t_first = None
    t_end = time.time() + seconds
    try:
        for frame in container.decode(video=0):
            now = time.time()
            if t_first is None:
                t_first = now
                result["first_frame_s"] = round(now - t0, 2)
                result["shape"] = [frame.height, frame.width]
            result["frames"] += 1
            if now > t_end:
                break
    except Exception as e:
        result["error"] = f"decode: {type(e).__name__}: {e}"
    container.close()
    if t_first is not None and result["frames"] > 1:
        result["fps"] = round((result["frames"] - 1) / max(time.time() - t_first, 1e-3), 1)
    return result


def child_cv2(seconds, open_timeout_ms, import_av_first):
    if import_av_first:
        import av  # noqa: F401  (djitellopy imports PyAV before reachglass opens OpenCV)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", REACHGLASS_CV2_OPTIONS)
    import cv2

    result = {"opened": False, "frames": 0, "open_timeout_ms": open_timeout_ms}
    t0 = time.time()
    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
                                                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    result["open_s"] = round(time.time() - t0, 2)
    if not cap.isOpened():
        result["error"] = f"VideoCapture did not open (after {result['open_s']} s)"
        return result
    result["opened"] = True
    t_first = None
    failed_reads = 0
    t_end = time.time() + seconds
    while time.time() < t_end and failed_reads < 3:
        ok, img = cap.read()
        if not ok:
            failed_reads += 1  # each failed read already waited up to the 3 s read timeout
            continue
        now = time.time()
        if t_first is None:
            t_first = now
            result["first_frame_s"] = round(now - t0, 2)
            result["shape"] = list(img.shape)
        result["frames"] += 1
        result["last_frame_mean"] = round(float(img.mean()), 1)  # ~0 = black frames
    cap.release()
    result["failed_reads"] = failed_reads
    if t_first is not None and result["frames"] > 1:
        result["fps"] = round((result["frames"] - 1) / max(time.time() - t_first, 1e-3), 1)
    return result


def child_reachglass(seconds):
    sys.path.insert(0, ROOT)
    result = {"opened": False, "frames": 0}
    try:
        import djitellopy  # noqa: F401  (the real app loads PyAV's FFmpeg this way)

        result["djitellopy_loaded"] = True
    except ImportError:
        result["djitellopy_loaded"] = False
    from reachglass.sources import TelloVideoSource

    t0 = time.time()
    source = TelloVideoSource(VIDEO_URL)
    try:
        source.start()
    except RuntimeError as e:
        result["open_s"] = round(time.time() - t0, 2)
        result["error"] = f"start(): {e}"
        return result
    result["opened"] = True
    result["open_s"] = round(time.time() - t0, 2)
    first = source.wait_first(10.0)
    if first is None:
        result["error"] = "opened, but no frame within 10 s"
        source.stop()
        return result
    result["first_frame_s"] = round(time.time() - t0, 2)
    result["shape"] = list(first.image.shape)
    last_seq = first.seq
    t_end = time.time() + seconds
    while time.time() < t_end:
        frame = source.read()
        if frame is not None and frame.seq != last_seq:
            result["frames"] += 1
            last_seq = frame.seq
        time.sleep(0.005)
    result["fps"] = round(result["frames"] / seconds, 1)
    result["reconnects"] = source.reconnects
    source.stop()
    return result


def run_child(name, seconds):
    cmd = [sys.executable, os.path.abspath(__file__), "--child", name, "--seconds", str(seconds)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 60)
    except subprocess.TimeoutExpired:
        return {"error": f"hung for more than {seconds + 60:.0f} s (killed)", "frames": 0}
    result = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if result is None:
        result = {"error": "no result", "frames": 0}
    result["returncode"] = p.returncode
    if p.returncode < 0:
        result["crash"] = signal.Signals(-p.returncode).name  # e.g. SIGSEGV, SIGTRAP, SIGABRT
    result["duplicate_ffmpeg_warning"] = "is implemented in both" in p.stderr
    result["stderr_tail"] = p.stderr.strip().splitlines()[-12:]
    return result


# ------------------------------------------------------------------ verdict
def works(decoder):
    return decoder is not None and decoder.get("frames", 0) > 5


def verdict(report):
    findings = []
    for port, status in report["ports"]["bind"].items():
        if status != "free":
            findings.append(f"UDP {port} is taken by another program ({status}). Find it with "
                            f"`lsof -nP -iUDP:{port}` and quit or kill it (e.g. gs-stand on 8889, a stale tello_io).")

    link = report.get("link")
    if link is None or link["reply"] is None:
        error = "" if link is None else str(link["error"])
        if "EHOSTUNREACH" in error or "No route" in error:
            findings.append("Sending to the drone fails with 'No route to host'. Either the laptop is not on TELLO-xxxxxx, "
                            "or (macOS 15+) Local Network privacy blocks the app that launched Python: System Settings > "
                            f"Privacy & Security > Local Network > enable '{report['launched_from']}', then quit and reopen it.")
        else:
            findings.append("The drone never answered 'command': laptop not on TELLO-xxxxxx, the phone app still "
                            "connected to the drone, or UDP 8889 taken/blocked.")
        return findings

    if report["state"]["packets"] == 0:
        findings.append("Commands work but no state packets arrive: inbound UDP to Python is blocked (macOS: allow "
                        "Python in System Settings > Network > Firewall; Windows: allow inbound UDP 8890 and 11111), "
                        "or UDP 8890 is taken.")

    video = report["video"]
    if video.get("packets", 0) == 0:
        findings.append("streamon was sent but NO video packets arrived on UDP 11111. Same causes as missing state "
                        "packets (firewall for incoming UDP, another program on 11111), or the phone app still owns the "
                        f"stream. streamon replied: {video.get('streamon_reply')!r}.")
        return findings
    if len(video["sps_s"]) == 0:
        findings.append(f"Video packets arrive but there was no SPS/PPS in {report['seconds']} s, so no decoder can start. "
                        "Power-cycle the Tello and rerun.")
    elif video["sps_s"][0] > 4.0:
        findings.append(f"The first keyframe header (SPS) came {video['sps_s'][0]} s after streamon. reachglass opens "
                        "OpenCV with a 5 s timeout and does not retry that first open, so it can give up before it.")

    decoders = report["decoders"]
    for name in DECODERS:
        d = decoders.get(name)
        if d is not None and "crash" in d:
            findings.append(f"Decoder '{name}' CRASHED ({d['crash']}). Its last stderr lines are in the report.")
    pyav = decoders.get("pyav")
    cv2_short = decoders.get("cv2")
    cv2_long = decoders.get("cv2_long_timeout")
    cv2_av = decoders.get("cv2_after_av")
    app = decoders.get("reachglass")

    if not works(cv2_short) and works(cv2_long):
        findings.append(f"OpenCV only opens the Tello stream with a longer timeout (took {cv2_long.get('open_s')} s): "
                        "reachglass's 5 s open timeout plus no retry on the first open is the failure.")
    if not works(cv2_short) and not works(cv2_long) and works(pyav):
        findings.append("OpenCV's FFmpeg cannot read the Tello stream on this machine, PyAV can: move reachglass's Tello "
                        "reader to PyAV (like flyfollow's VideoReader).")
    if works(cv2_short) and not works(cv2_av):
        findings.append("OpenCV reads the stream alone but fails once PyAV is loaded in the same process (two FFmpeg "
                        "builds, one from cv2 and one from av/djitellopy): keep one decoder per process.")
    if works(cv2_short) and works(cv2_av) and not works(app):
        findings.append("Plain OpenCV works but reachglass.TelloVideoSource does not: the bug is in reachglass's reader "
                        "(sources/video_stream.py, sources/opencv_sources.py).")
    if not works(pyav) and not works(cv2_short) and not works(cv2_long):
        findings.append("Raw video arrives but no decoder produced frames. Check the stderr lines of each decoder in "
                        "the report.")
    if len(findings) == 0:
        findings.append("Every layer works here, including reachglass's own reader. The failure is then in the app "
                        "run itself (order of startup, another program starting at the same time, a different venv): "
                        "rerun the app right after this and keep its full output.")
    return findings


# ------------------------------------------------------------------ main
def run(args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launched_from = os.environ.get("__CFBundleIdentifier", os.environ.get("TERM_PROGRAM", "your terminal app"))
    report = {"stamp": stamp, "ip": args.ip, "seconds": args.seconds, "platform": platform.platform(),
              "python": sys.executable, "launched_from": launched_from, "decoders": {}}
    print(f"Tello video diagnosis | {report['platform']} | {sys.executable} | launched from {launched_from}")

    info = run_child("info", args.seconds)
    report["packages"] = info
    opencvs = []
    for name in OPENCV_PACKAGES:
        if name in info.get("packages", {}):
            opencvs.append(f"{name} {info['packages'][name]}")
    say("info", "packages", f"{', '.join(opencvs)} | av {info.get('packages', {}).get('av')} "
                            f"(libavformat {info.get('av_libavformat')}) | djitellopy {info.get('packages', {}).get('djitellopy')}")
    say("info", "cv2 FFmpeg", " / ".join(info.get("cv2_ffmpeg", [info.get("cv2_error", "?")])))
    if len(opencvs) > 1:
        say("WARN", "packages", "more than one OpenCV package installed: they overwrite each other's cv2 folder")
    if info.get("duplicate_ffmpeg_warning"):
        say("WARN", "packages", "cv2 and av both load their own FFmpeg (macOS 'Class ... is implemented in both')")

    ports = check_ports()
    report["ports"] = ports
    for port, status in ports["bind"].items():
        say("OK" if status == "free" else "FAIL", f"port {port}", status)
    if ports["lsof"]:
        say("FAIL", "port holders", ports["lsof"].replace("\n", "\n" + " " * 25))

    network = check_network(args.ip)
    report["network"] = network
    local_ip = network["local_ip"]
    on_tello = local_ip is not None and local_ip.startswith("192.168.10.")
    say("OK" if on_tello or args.ip != "192.168.10.1" else "WARN", "laptop IP",
        f"{local_ip} towards {args.ip}" + ("" if on_tello else "  (expected 192.168.10.x on TELLO-xxxxxx)"))
    if network["route"]:
        say("info", "route", " | ".join(network["route"]))
    if network["firewall"]:
        firewall = []
        for line in network["firewall"]:
            if line:
                firewall.append(line)
        say("info", "macOS firewall", " | ".join(firewall))

    state_sock, state_error = open_udp(STATE_PORT)
    video_sock, video_error = open_udp(VIDEO_PORT)
    link_sock, link_error = open_udp(0)
    link = {"reply": None, "error": link_error, "rtt_s": None}
    for i in range(3):
        reply, rtt, error = send(link_sock, args.ip, "command")
        link = {"reply": reply, "error": error, "rtt_s": round(rtt, 3)}
        if reply is not None:
            break
    report["link"] = link
    if link["reply"] is None:
        say("FAIL", "link", f"'command' got no reply: {link['error']}")
    else:
        say("OK" if "ok" in link["reply"].lower() else "WARN", "link", f"'command' -> {link['reply']!r} in {link['rtt_s']} s")

    if link["reply"] is not None:
        if state_sock is None:
            report["state"] = {"packets": 0, "error": state_error}
            say("FAIL", "state", f"cannot listen on UDP {STATE_PORT}: {state_error}")
        else:
            report["state"] = watch_state(state_sock, 2.0)
            say("OK" if report["state"]["packets"] > 0 else "FAIL", "state",
                f"{report['state']['packets']} packets in 2 s, battery {report['state']['battery']}%")

        if video_sock is None:
            report["video"] = {"packets": 0, "error": video_error}
            reply, rtt, error = send(link_sock, args.ip, "streamon", timeout=7.0)
            report["video"]["streamon_reply"] = reply
            say("FAIL", "raw video", f"cannot listen on UDP {VIDEO_PORT}: {video_error}")
        else:
            already = drain(video_sock, 0.5)
            t0 = time.time()
            reply, rtt, error = send(link_sock, args.ip, "streamon", timeout=7.0)
            report["video"] = watch_video(video_sock, args.seconds, t0)
            report["video"]["streamon_reply"] = reply if reply is not None else error
            report["video"]["packets_before_streamon"] = already
            video = report["video"]
            sps = video["sps_s"]
            gaps = []
            for i in range(1, len(sps)):
                gaps.append(sps[i] - sps[i - 1])
            spacing = f"every {sum(gaps) / len(gaps):.1f} s" if len(gaps) > 0 else "once or never"
            say("OK" if video["packets"] > 0 else "FAIL", "raw video",
                f"streamon -> {video['streamon_reply']!r} | {video['packets']} packets, {video['kbit_s']} kbit/s from "
                f"{', '.join(video['senders']) if video['senders'] else 'nobody'} | first packet {video['first_packet_s']} s")
            say("OK" if len(sps) > 0 else "FAIL", "keyframes",
                f"NAL counts {video['nal_counts']} | first SPS at {sps[0] if sps else None} s, {spacing}")

        for sock in [state_sock, video_sock]:
            if sock is not None:
                sock.close()

        for name in DECODERS:
            send(link_sock, args.ip, "battery?")  # keep the SDK session alive between decoders
            print(f"         {name}: running...", flush=True)
            d = run_child(name, args.seconds)
            report["decoders"][name] = d
            detail = f"opened {d.get('opened')} in {d.get('open_s')} s | first frame {d.get('first_frame_s')} s | " \
                     f"{d.get('frames')} frames, {d.get('fps')} fps, shape {d.get('shape')}"
            if "error" in d:
                detail += f" | {d['error']}"
            if "crash" in d:
                detail += f" | CRASH {d['crash']}"
            say("OK" if works(d) else "FAIL", name, detail)

        send(link_sock, args.ip, "streamoff")
    else:
        report["state"] = {"packets": 0}
        report["video"] = {"packets": 0}
        for sock in [state_sock, video_sock]:
            if sock is not None:
                sock.close()
    if link_sock is not None:
        link_sock.close()

    report["verdict"] = verdict(report)
    print("\nVERDICT")
    for line in report["verdict"]:
        print(f"  - {line}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"tello_video_diag_{stamp}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nfull report -> {path}")


def main():
    p = argparse.ArgumentParser(description="Find where the Tello video chain breaks on this laptop (no flight).")
    p.add_argument("--ip", default="192.168.10.1", help="Tello address (default 192.168.10.1)")
    p.add_argument("--seconds", type=float, default=6.0, help="how long to watch the stream per stage (default 6)")
    p.add_argument("--child", choices=["info"] + DECODERS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child is None:
        run(args)
        return

    if args.child == "info":
        result = child_info(args.seconds)
    elif args.child == "pyav":
        result = child_pyav(args.seconds)
    elif args.child == "cv2":
        result = child_cv2(args.seconds, 5000, False)
    elif args.child == "cv2_long_timeout":
        result = child_cv2(args.seconds, 30000, False)
    elif args.child == "cv2_after_av":
        result = child_cv2(args.seconds, 5000, True)
    else:
        result = child_reachglass(args.seconds)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
