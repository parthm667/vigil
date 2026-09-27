"""Runtime message bus: ZeroMQ broker, JSON pub/sub, an in-process twin for tests, and the shared-memory frame ring.

Contract: flyfollow/runtime/messages.py. Every message is one JSON object sent as a two-frame multipart
[topic bytes, json bytes]. ZeroMQ filters subscriptions by PREFIX ("det" also matches "det_cfg"), so the
Subscriber re-checks the exact topic name after receiving.

Addresses: pass nothing and the defaults are messages.PUB_ADDR / SUB_ADDR / FRAME_RING_NAME, unless the
environment sets FLYFOLLOW_PUB_ADDR, FLYFOLLOW_SUB_ADDR or FLYFOLLOW_FRAME_RING (the launcher uses these to
run tests and parallel sessions on other ports). Code that passes the constants explicitly bypasses that.

Slow joiner: a PUB/SUB connection is asynchronous, so the first messages after a Publisher or Subscriber is
created can be lost. Both constructors wait (up to wait_s, default 0.5 s) for the ZeroMQ handshake with the
broker plus a short settle for subscriptions to propagate. Streams (tello_state, rc, det) do not care; for
one-shot messages at startup, publish after construction returns, or repeat them.

Ordering: messages from one Publisher arrive in order. There is no global order across publishers: sort by "t".

    python -m flyfollow.runtime.bus                  # run a broker in the foreground
    python -m flyfollow.runtime.bus --echo det,rc    # print messages (all topics if no list)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import zmq

from flyfollow.runtime.messages import FRAME_RING_NAME, FRAME_RING_SLOTS, PUB_ADDR, SUB_ADDR

HWM = 10_000  # per-socket queue limit; beyond it ZeroMQ PUB/XPUB drops (never blocks)
SETTLE_S = 0.03  # after the handshake, time for subscriptions to reach the publishers


def default_pub_addr() -> str:
    return os.environ.get("FLYFOLLOW_PUB_ADDR", PUB_ADDR)


def default_sub_addr() -> str:
    return os.environ.get("FLYFOLLOW_SUB_ADDR", SUB_ADDR)


def default_frame_ring() -> str:
    return os.environ.get("FLYFOLLOW_FRAME_RING", FRAME_RING_NAME)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def encode(m: dict) -> bytes:
    return json.dumps(m, default=_json_default, separators=(",", ":")).encode()


def decode(b: bytes) -> dict:
    return json.loads(b)


def _prepare(m: dict, name: str) -> dict:
    if "topic" not in m:
        raise ValueError(f"message has no topic: {m!r}")
    out = dict(m)
    out.setdefault("t", time.time())
    out.setdefault("src_node", name)
    return out


def _wait_handshake(sock: zmq.Socket, wait_s: float) -> bool:
    """Block until the socket completes a ZMTP handshake with its peer (or wait_s passes)."""
    if wait_s <= 0:
        return False
    mon = sock.get_monitor_socket(zmq.EVENT_HANDSHAKE_SUCCEEDED)
    try:
        ok = bool(mon.poll(int(wait_s * 1000)))
        if ok:
            mon.recv_multipart()
            time.sleep(SETTLE_S)
        return ok
    finally:
        sock.disable_monitor()
        mon.close(linger=0)


# ------------------------------------------------------------------------------------------------ broker
class Broker:
    """XSUB (bind pub_addr) <-> XPUB (bind sub_addr) proxy in a daemon thread. Stops cleanly via a control socket."""

    def __init__(self, pub_addr: str | None = None, sub_addr: str | None = None):
        self.pub_addr = pub_addr or default_pub_addr()
        self.sub_addr = sub_addr or default_sub_addr()
        self._ctx = zmq.Context.instance()
        self._thread: threading.Thread | None = None
        self._ctrl: zmq.Socket | None = None

    def start(self) -> "Broker":
        ctrl_addr = f"inproc://broker-ctrl-{uuid.uuid4().hex[:8]}"
        xsub = self._ctx.socket(zmq.XSUB)
        xpub = self._ctx.socket(zmq.XPUB)
        for s in (xsub, xpub):
            s.linger = 0
            s.sndhwm = HWM
            s.rcvhwm = HWM
        try:
            xsub.bind(self.pub_addr)
            xpub.bind(self.sub_addr)
        except zmq.ZMQError as e:
            xsub.close()
            xpub.close()
            raise RuntimeError(
                f"broker cannot bind {self.pub_addr} / {self.sub_addr} ({e}). Another broker or a crashed run may hold "
                f"the port: lsof -nP -iTCP:{self.pub_addr.rsplit(':', 1)[-1]} and kill it."
            ) from e
        ctrl_in = self._ctx.socket(zmq.PAIR)
        ctrl_in.bind(ctrl_addr)
        self._ctrl = self._ctx.socket(zmq.PAIR)
        self._ctrl.linger = 0
        self._ctrl.connect(ctrl_addr)

        def run() -> None:  # sockets move to this thread; the main thread no longer touches them
            try:
                zmq.proxy_steerable(xsub, xpub, None, ctrl_in)
            except zmq.ContextTerminated:
                pass
            finally:
                for s in (xsub, xpub, ctrl_in):
                    s.close(linger=0)

        self._thread = threading.Thread(target=run, name="bus-broker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self._ctrl.send(b"TERMINATE")
        except zmq.ZMQError:
            pass
        self._thread.join(timeout=2.0)
        self._ctrl.close(linger=0)
        self._thread = None

    def __enter__(self) -> "Broker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ------------------------------------------------------------------------------------------------ in-process bus
class _InProcQueue:
    def __init__(self, bus: "InProcBus", topics: list[str] | None, conflate: bool):
        self.bus = bus
        self.topics = set(topics) if topics else None
        self.conflate = conflate
        self.q: collections.deque[dict] = collections.deque(maxlen=HWM)

    def wants(self, topic: str) -> bool:
        return self.topics is None or topic in self.topics


class InProcBus:
    """Same Publisher/Subscriber semantics without sockets (thread-safe, one process). Messages are JSON round-tripped."""

    def __init__(self):
        self._cv = threading.Condition()
        self._subs: list[_InProcQueue] = []

    def _add(self, q: _InProcQueue) -> None:
        with self._cv:
            self._subs.append(q)

    def _remove(self, q: _InProcQueue) -> None:
        with self._cv:
            if q in self._subs:
                self._subs.remove(q)

    def _deliver(self, m: dict) -> None:
        data = encode(m)
        with self._cv:
            for q in self._subs:
                if q.wants(m["topic"]):
                    q.q.append(decode(data))
            self._cv.notify_all()

    def _pop(self, q: _InProcQueue, timeout_s: float) -> dict | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cv:
            while not q.q:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return q.q.popleft()


# ------------------------------------------------------------------------------------------------ publisher
class Publisher:
    """Never blocks: NOBLOCK send, messages dropped at the high-water mark. Adds src_node and t when absent."""

    def __init__(self, name: str, addr: str | None = None, bus: InProcBus | None = None, wait_s: float = 0.5):
        self.name = name
        self.bus = bus
        self.sock: zmq.Socket | None = None
        self.dropped = 0
        if bus is None:
            self.sock = zmq.Context.instance().socket(zmq.PUB)
            self.sock.linger = 200  # let queued messages (a final land) leave on close
            self.sock.sndhwm = HWM
            self.sock.connect(addr or default_pub_addr())
            self.connected = _wait_handshake(self.sock, wait_s)
        else:
            self.connected = True

    def publish(self, m: dict) -> None:
        m = _prepare(m, self.name)
        if self.bus is not None:
            self.bus._deliver(m)
            return
        try:
            self.sock.send_multipart([m["topic"].encode(), encode(m)], flags=zmq.NOBLOCK)
        except zmq.Again:
            self.dropped += 1

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self) -> "Publisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------------------------------------ subscriber
class Subscriber:
    """Exact-topic subscriber (topics None = everything). conflate=True keeps only the newest message per topic."""

    def __init__(self, topics: list[str] | None = None, addr: str | None = None, bus: InProcBus | None = None,
                 conflate: bool = False, wait_s: float = 0.5):
        self.topics = set(topics) if topics else None
        self.conflate = conflate
        self.bus = bus
        self.sock: zmq.Socket | None = None
        self._pending: dict[str, dict] = {}  # conflate buffer, newest per topic
        self._q: _InProcQueue | None = None
        if bus is not None:
            self._q = _InProcQueue(bus, topics, conflate)
            bus._add(self._q)
            self.connected = True
            return
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.linger = 0
        self.sock.rcvhwm = HWM
        for tp in topics or [""]:
            self.sock.setsockopt(zmq.SUBSCRIBE, tp.encode())
        self.sock.connect(addr or default_sub_addr())
        self.connected = _wait_handshake(self.sock, wait_s)

    def _wants(self, topic: str) -> bool:
        return self.topics is None or topic in self.topics

    def _recv_one(self, timeout_s: float) -> dict | None:
        """Next matching message from the transport (no conflation)."""
        if self._q is not None:
            return self.bus._pop(self._q, timeout_s)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            left_ms = max(0, int(math.ceil((deadline - time.monotonic()) * 1000)))
            if not self.sock.poll(left_ms):
                return None
            try:
                parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            if len(parts) != 2:
                continue
            topic = parts[0].decode(errors="replace")
            if not self._wants(topic):
                continue  # prefix match only ("det" vs "det_cfg")
            try:
                return decode(parts[1])
            except ValueError:
                continue

    def recv(self, timeout_s: float = 0.0) -> dict | None:
        """Next message or None. With conflate, the oldest of the newest-per-topic messages."""
        if not self.conflate:
            return self._recv_one(timeout_s)
        self._fill(max_n=HWM)
        if not self._pending and timeout_s > 0:
            m = self._recv_one(timeout_s)
            if m is not None:
                self._pending[m["topic"]] = m
                self._fill(max_n=HWM)
        if not self._pending:
            return None
        topic = min(self._pending, key=lambda k: self._pending[k].get("t", 0.0))
        return self._pending.pop(topic)

    def _fill(self, max_n: int) -> None:
        for _ in range(max_n):
            m = self._recv_one(0.0)
            if m is None:
                return
            self._pending[m["topic"]] = m

    def drain(self, max_n: int = 1000) -> list[dict]:
        """Everything pending, non-blocking (conflate: newest per topic, oldest first)."""
        if self.conflate:
            self._fill(max_n)
            out = sorted(self._pending.values(), key=lambda m: m.get("t", 0.0))
            self._pending.clear()
            return out
        out = []
        for _ in range(max_n):
            m = self._recv_one(0.0)
            if m is None:
                break
            out.append(m)
        return out

    def close(self) -> None:
        if self._q is not None:
            self.bus._remove(self._q)
            self._q = None
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None

    def __enter__(self) -> "Subscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------------------------------------ frame ring
_MAGIC = 0x46524E47  # "FRNG"
_GHDR = 64  # global header bytes: int64 [magic, slots, h, w, write_count, ...]
_SHDR = 32  # per-slot header bytes: int64 seq (odd while writing), int64 frame_id, float64 t_decoded, pad
_SHM_MAX = 30  # macOS PSHMNAMLEN is 31 including the leading "/"
_attach_lock = threading.Lock()


def shm_name(name: str) -> str:
    """Shorten a ring name to the macOS POSIX shared-memory limit (deterministic, so both sides agree)."""
    name = name.lstrip("/")
    if len(name) <= _SHM_MAX:
        return name
    return name[: _SHM_MAX - 9] + "_" + hashlib.sha1(name.encode()).hexdigest()[:8]


def _open_untracked(name: str) -> shared_memory.SharedMemory:
    """Attach without registering with the resource tracker (Python < 3.13 would unlink the segment and warn
    'leaked shared_memory objects' when a consumer exits)."""
    with _attach_lock:
        orig = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = orig


def _unlink_raw(name: str) -> None:
    try:
        import _posixshmem

        _posixshmem.shm_unlink("/" + name)
    except (ImportError, FileNotFoundError):
        pass


class FrameRing:
    """multiprocessing.shared_memory ring of RGB uint8 frames. One producer (Tello I/O, sim or replay), many readers.

    Readers copy a slot out and get None if the producer overwrote it meanwhile (per-slot seqlock + frame_id).
    """

    def __init__(self, shm: shared_memory.SharedMemory, owner: bool):
        self.shm = shm
        self.owner = owner
        self.name = shm.name.lstrip("/")
        g = np.ndarray((_GHDR // 8,), dtype=np.int64, buffer=shm.buf, offset=0)
        if g[0] != _MAGIC:
            raise RuntimeError(f"shared memory {self.name!r} is not a FrameRing")
        self._g = g
        self.slots, self.h, self.w = int(g[1]), int(g[2]), int(g[3])
        fb = self.h * self.w * 3
        self._stride = _SHDR + fb
        self._hdr_i = []
        self._hdr_f = []
        self._img = []
        for i in range(self.slots):
            off = _GHDR + i * self._stride
            self._hdr_i.append(np.ndarray((4,), dtype=np.int64, buffer=shm.buf, offset=off))
            self._hdr_f.append(np.ndarray((4,), dtype=np.float64, buffer=shm.buf, offset=off))
            self._img.append(np.ndarray((self.h, self.w, 3), dtype=np.uint8, buffer=shm.buf, offset=off + _SHDR))

    @classmethod
    def create(cls, name: str | None = None, slots: int = FRAME_RING_SLOTS, h: int = 720, w: int = 960) -> "FrameRing":
        """Create (producer). A same-named segment left by a crashed run is unlinked and recreated."""
        n = shm_name(name or default_frame_ring())
        size = _GHDR + slots * (_SHDR + h * w * 3)
        try:
            shm = shared_memory.SharedMemory(name=n, create=True, size=size)
        except FileExistsError:
            _unlink_raw(n)
            shm = shared_memory.SharedMemory(name=n, create=True, size=size)
        g = np.ndarray((_GHDR // 8,), dtype=np.int64, buffer=shm.buf, offset=0)
        g[:] = 0
        g[1:4] = (slots, h, w)
        for i in range(slots):  # frame_id -1 = empty
            np.ndarray((4,), dtype=np.int64, buffer=shm.buf, offset=_GHDR + i * (_SHDR + h * w * 3))[:] = (0, -1, 0, 0)
        g[0] = _MAGIC
        del g
        return cls(shm, owner=True)

    @classmethod
    def attach(cls, name: str | None = None, timeout_s: float = 0.0) -> "FrameRing":
        """Attach (consumer). Retries until timeout_s if the producer has not created the ring yet."""
        n = shm_name(name or default_frame_ring())
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return cls(_open_untracked(n), owner=False)
            except (FileNotFoundError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def write(self, frame, frame_id: int, t_decoded: float) -> int:
        """Copy one (h, w, 3) uint8 RGB frame into the next slot; returns the slot."""
        frame = np.asarray(frame)
        if frame.shape != (self.h, self.w, 3):
            raise ValueError(f"frame shape {frame.shape} != ring shape {(self.h, self.w, 3)}")
        slot = int(self._g[4] % self.slots)
        hi, hf = self._hdr_i[slot], self._hdr_f[slot]
        hi[0] += 1  # odd: writing
        hi[1] = -1
        self._img[slot][...] = frame
        hf[2] = float(t_decoded)
        hi[1] = int(frame_id)
        hi[0] += 1  # even: stable
        self._g[4] += 1
        return slot

    def read(self, slot: int, frame_id: int):
        """Copy of the frame in slot, or None if it no longer holds frame_id (overwritten or being written)."""
        if not 0 <= slot < self.slots:
            return None
        hi = self._hdr_i[slot]
        s1 = int(hi[0])
        if s1 % 2 or int(hi[1]) != frame_id:
            return None
        out = self._img[slot].copy()
        if int(hi[0]) != s1 or int(hi[1]) != frame_id:
            return None
        return out

    def latest(self):
        """(frame copy, frame_id, t_decoded) of the newest complete frame, or None."""
        for _ in range(3):
            n = int(self._g[4])
            if n == 0:
                return None
            slot = (n - 1) % self.slots
            hi, hf = self._hdr_i[slot], self._hdr_f[slot]
            s1 = int(hi[0])
            fid = int(hi[1])
            t = float(hf[2])
            if s1 % 2 or fid < 0:
                continue
            img = self.read(slot, fid)
            if img is not None and int(hi[0]) == s1:
                return img, fid, t
        return None

    @property
    def write_count(self) -> int:
        return int(self._g[4])

    def close(self) -> None:
        if self.shm is None:
            return
        self._img = self._hdr_i = self._hdr_f = []
        self._g = None
        try:
            self.shm.close()
        except BufferError:
            pass  # a caller still holds a view; the OS frees the mapping at exit
        self._closed, self.shm = self.shm, None

    def unlink(self) -> None:
        """Remove the segment (producer only; readers keep their mapping until they close)."""
        shm = self.shm or getattr(self, "_closed", None)
        if shm is None or not self.owner:
            return
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "FrameRing":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
        if self.owner:
            self.unlink()


# ------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="flyfollow runtime broker / bus echo")
    ap.add_argument("--echo", nargs="?", const="", default=None, help="print messages (comma-separated topics, default all)")
    ap.add_argument("--no-broker", action="store_true", help="with --echo: attach to a running broker")
    a = ap.parse_args(argv)
    broker = None
    if a.echo is None or not a.no_broker:
        broker = Broker().start()
        print(f"[bus] broker {broker.pub_addr} -> {broker.sub_addr}", flush=True)
    try:
        if a.echo is not None:
            sub = Subscriber([t for t in a.echo.split(",") if t] or None)
            while True:
                m = sub.recv(0.5)
                if m is not None:
                    print(json.dumps(m, separators=(",", ":")), flush=True)
        else:
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        if broker is not None:
            broker.stop()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
