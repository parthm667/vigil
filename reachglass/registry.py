"""Name -> factory registries: how config picks an implementation for each pluggable part.

    DETECTORS = Registry("detector")

    @DETECTORS.register("color_blob")
    def _make(**params): return ColorBlobDetector(**params)

    det = DETECTORS.build(cfg.perception.target_detector)   # ComponentSpec(kind, params)
"""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, what: str):
        self.what = what
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(factory):
            if name in self._factories:
                raise ValueError(f"{self.what} '{name}' registered twice")
            self._factories[name] = factory
            return factory

        return deco

    def names(self) -> list[str]:
        return sorted(self._factories)

    def build(self, spec: Any = None, /, **overrides) -> Any:
        """spec: a ComponentSpec, a kind string, or None/'' (returns None: component disabled)."""
        if spec is None or spec == "":
            return None
        if isinstance(spec, str):
            kind, params = spec, {}
        else:
            kind, params = spec.kind, dict(spec.params)
        if not kind:
            return None
        if kind not in self._factories:
            raise KeyError(f"unknown {self.what} '{kind}' (have: {', '.join(self.names())})")
        params.update(overrides)
        return self._factories[kind](**params)
