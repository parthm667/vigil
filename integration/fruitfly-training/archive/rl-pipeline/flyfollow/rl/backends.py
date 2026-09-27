"""Where evaluation jobs run. Every backend has map(jobs) -> results in the same order.

serial: this process (tests, debugging)
local:  a process pool on this machine's CPU cores
modal:  Modal CPU containers (flyfollow/rl/modal_app.py), same job function
"""

from __future__ import annotations

import multiprocessing

from .worker import run_job


class SerialBackend:
    name = "serial"

    def map(self, jobs: list[dict]) -> list[dict]:
        results = []
        for job in jobs:
            results.append(run_job(job))
        return results

    def close(self) -> None:
        pass


class LocalBackend:
    name = "local"

    def __init__(self, workers: int):
        self.workers = workers
        self.pool = multiprocessing.get_context("spawn").Pool(workers)

    def map(self, jobs: list[dict]) -> list[dict]:
        return self.pool.map(run_job, jobs, chunksize=1)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


class ModalBackend:
    name = "modal"

    def __init__(self, evaluate_fn):
        # evaluate_fn is the Modal Function wrapping run_job (see modal_app.py)
        self.evaluate = evaluate_fn

    def map(self, jobs: list[dict]) -> list[dict]:
        return list(self.evaluate.map(jobs, order_outputs=True))

    def close(self) -> None:
        pass


def make_backend(name: str, workers: int = 4, evaluate_fn=None):
    if name == "serial":
        return SerialBackend()
    if name == "local":
        return LocalBackend(workers)
    if name == "modal":
        if evaluate_fn is None:
            raise ValueError("the modal backend is created inside modal_app.py")
        return ModalBackend(evaluate_fn)
    raise ValueError(f"unknown backend {name}")
