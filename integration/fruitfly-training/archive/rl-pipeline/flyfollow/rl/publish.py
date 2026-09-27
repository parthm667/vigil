"""Publish training checkpoints to the `rl-results` branch from a separate git worktree.

Rules (so auto-commits never collide with code work):
- results live on their own branch (`rl-results`), checked out in its own worktree,
  never on the code branch or in the code working copy;
- only the checkpoint file and HANDOFF.md are staged, never `git add -A`;
- a failed push (no Wi-Fi, laptop on the Tello network) is logged and retried at the
  next push; it never stops training.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

STATUS_BEGIN = "<!-- STATUS:BEGIN -->"
STATUS_END = "<!-- STATUS:END -->"
LOCK_NAME = ".publish.lock"


class Publisher:
    def __init__(self, worktree: str | Path, branch: str = "rl-results", remote: str = "origin", log_path: str | Path | None = None):
        self.worktree = Path(worktree).resolve()
        self.branch = branch
        self.remote = remote
        self.log_path = Path(log_path) if log_path else None

    # ------------------------------------------------------------------ logging
    def log(self, msg: str) -> None:
        line = f"[publish {datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def ready(self) -> bool:
        if not (self.worktree / ".git").exists():
            self.log(f"results worktree {self.worktree} not found; skipping publish (see HANDOFF.md setup)")
            return False
        return True

    # ------------------------------------------------------------------ files
    def write_checkpoint(self, folder: str, data: dict) -> Path:
        path = self.worktree / "checkpoints" / folder / "latest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def update_handoff(self) -> Path:
        path = self.worktree / "HANDOFF.md"
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = f"# Training results\n\n{STATUS_BEGIN}\n{STATUS_END}\n"
        block = self.status_block()
        start = text.find(STATUS_BEGIN)
        end = text.find(STATUS_END)
        if start == -1 or end == -1:
            text = text + f"\n{STATUS_BEGIN}\n{block}\n{STATUS_END}\n"
        else:
            text = text[: start + len(STATUS_BEGIN)] + "\n" + block + "\n" + text[end:]
        path.write_text(text, encoding="utf-8")
        return path

    def status_block(self) -> str:
        rows = []
        root = self.worktree / "checkpoints"
        if root.exists():
            for folder in sorted(root.iterdir()):
                f = folder / "latest.json"
                if not f.exists():
                    continue
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                best = d.get("best_selection_score")
                best_text = "n/a" if best is None else f"{best:.3f} (gen {d.get('best_generation')})"
                rows.append(
                    f"| {folder.name} | {d.get('run_name')} | {d.get('backend')} | {d.get('generation')} | {best_text} | "
                    f"{d.get('latest_train_score', float('nan')):.3f} | {d.get('sigma', float('nan')):.4f} | "
                    f"{'finished' if d.get('finished') else 'running'} | {d.get('updated_utc')} |"
                )
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"_Last updated {now}. Written automatically by the trainer (`--push-every`)._",
            "",
            "Score = episode return divided by |return of the hand-tuned PID| on the same seed, averaged half follow, half approach. "
            "About -1.0 means as good as the hand-tuned PID; higher is better.",
            "",
            "| Checkpoint | Run | Backend | Generation | Best selection score | Latest train score | Sigma | State | Updated |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        if rows:
            lines.extend(rows)
        else:
            lines.append("| (none yet) | | | | | | | | |")
        return "\n".join(lines)

    # ------------------------------------------------------------------ git
    def _git(self, *args, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.worktree, capture_output=True, text=True, timeout=timeout)

    def _acquire(self, wait_s: float = 120.0) -> bool:
        lock = self.worktree / LOCK_NAME
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 300:
                        lock.unlink()  # stale lock from a killed process
                        continue
                except OSError:
                    pass
                time.sleep(1.0)
        return False

    def _release(self) -> None:
        try:
            (self.worktree / LOCK_NAME).unlink()
        except OSError:
            pass

    def _commit_and_push(self, paths: list[Path], message: str) -> bool:
        """Stage only `paths`, commit if anything changed, push. Caller holds the lock."""
        rel = []
        for p in paths:
            rel.append(str(Path(p).resolve().relative_to(self.worktree)).replace("\\", "/"))
        r = self._git("add", "--", *rel)
        if r.returncode != 0:
            self.log(f"git add failed: {r.stderr.strip()}")
            return False
        staged = self._git("diff", "--cached", "--quiet", "--", *rel)
        if staged.returncode == 1:
            r = self._git("commit", "-m", message, "--", *rel)
            if r.returncode != 0:
                self.log(f"git commit failed: {r.stderr.strip() or r.stdout.strip()}")
                return False
        return self._push()

    def _push(self) -> bool:
        ahead = self._git("rev-list", "--count", f"{self.remote}/{self.branch}..HEAD")
        if ahead.returncode == 0 and ahead.stdout.strip() == "0":
            return True
        r = self._git("push", self.remote, f"HEAD:{self.branch}", timeout=90)
        if r.returncode == 0:
            self.log(f"pushed to {self.remote}/{self.branch}")
            return True
        self.log(f"push failed ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no message'}); trying pull --rebase")
        pull = self._git("pull", "--rebase", self.remote, self.branch, timeout=90)
        if pull.returncode != 0:
            self._git("rebase", "--abort")
            self.log(f"pull failed: {pull.stderr.strip()[:200]}; will retry next push")
            return False
        r = self._git("push", self.remote, f"HEAD:{self.branch}", timeout=90)
        if r.returncode == 0:
            self.log(f"pushed to {self.remote}/{self.branch} after rebase")
            return True
        self.log(f"push failed again: {r.stderr.strip()[:200]}; will retry next push")
        return False

    def publish(self, folder: str, data: dict, message: str) -> bool:
        """Write the checkpoint and status block, then commit and push only those two files. Never raises."""
        if not self.ready():
            return False
        if not self._acquire():
            self.log("could not get the publish lock; will retry next time")
            return False
        try:
            ckpt = self.write_checkpoint(folder, data)
            handoff = self.update_handoff()
            return self._commit_and_push([ckpt, handoff], message)
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            self.log(f"publish error: {e}")
            return False
        finally:
            self._release()
