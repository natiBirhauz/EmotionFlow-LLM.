"""Performance monitoring utilities for pipeline component timing."""

import time
from contextlib import contextmanager
from typing import Dict, List


class PerformanceMonitor:
    """Track execution time for pipeline components."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager that records elapsed time in seconds under name."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)

    def report(self) -> Dict[str, Dict[str, float]]:
        """Return a dict mapping component names to {mean, min, max, total} stats."""
        stats = {}
        for component, times in self.timings.items():
            if times:
                stats[component] = {
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                }
        return stats
