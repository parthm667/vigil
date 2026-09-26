"""Where text queries come from. All inboxes: poll(now) -> list of new query strings.

  StdinInbox      type a sentence in the terminal and press Enter
  UdpInbox        other programs (the voice/STT team) send UTF-8 datagrams:
                    python -c "import socket; socket.socket(2,2).sendto(b'find my water bottle', ('127.0.0.1', 5005))"
  ScriptedInbox   [(time, text), ...] for simulations and tests
  MultiInbox      several at once
"""

from __future__ import annotations

import queue
import socket
import sys
import threading


class ScriptedInbox:
    def __init__(self, items: list[tuple[float, str]]):
        self.items = sorted(items)

    def poll(self, now: float) -> list[str]:
        out = [text for t, text in self.items if t <= now]
        self.items = [(t, text) for t, text in self.items if t > now]
        return out

    def push(self, now: float, text: str) -> None:
        self.items.append((now, text))
        self.items.sort()


class StdinInbox:
    def __init__(self, stream=None):
        self.q: queue.Queue = queue.Queue()
        self.stream = stream or sys.stdin
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.stream:
            line = line.strip()
            if line:
                self.q.put(line)

    def poll(self, now: float) -> list[str]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out


class UdpInbox:
    def __init__(self, port: int = 5005, host: str = "127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]

    def poll(self, now: float) -> list[str]:
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            text = data.decode("utf-8", errors="ignore").strip()
            if text:
                out.append(text)
        return out

    def close(self) -> None:
        self.sock.close()


class MultiInbox:
    def __init__(self, *inboxes):
        self.inboxes = [i for i in inboxes if i is not None]

    def poll(self, now: float) -> list[str]:
        return [t for i in self.inboxes for t in i.poll(now)]
