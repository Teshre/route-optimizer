"""
Shared :class:`Solution` container used by both the OR-Tools solver and the
baseline heuristic, so :mod:`route_optimizer.kpis` can compare them on equal
footing.

A ``Solution`` describes a complete assignment of clients to vehicles:

* ``routes[v]`` — the ordered list of **DataFrame row-indices** visited by
  vehicle ``v`` (the depot is implicit and excluded).
* ``per_vehicle[v]`` — a per-route summary dict.
* global totals + the list of ``dropped`` client indices + which distance
  provider was actually used + an overall ``feasible`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Solution:
    """A routing solution shared across solver / baseline / kpis."""

    routes: list[list[int]] = field(default_factory=list)
    per_vehicle: list[dict] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_time_min: float = 0.0
    dropped: list[int] = field(default_factory=list)
    provider_used: str = "haversine"
    feasible: bool = False

    @property
    def num_vehicles_used(self) -> int:
        """How many vehicles actually serve at least one client."""
        return sum(1 for r in self.routes if r)

    @property
    def num_served(self) -> int:
        """Total number of clients served across all routes."""
        return sum(len(r) for r in self.routes)
