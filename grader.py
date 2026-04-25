from __future__ import annotations

from typing import Any


def scalar_final_score(trajectory: dict[str, Any]) -> float:
    return float(trajectory.get("terminal_score", 0.0))


def bankruptcy_aware(trajectory: dict[str, Any]) -> float:
    score = scalar_final_score(trajectory)
    if trajectory.get("ever_bankrupt"):
        return score * 0.5
    return score


def stewardship_weighted(trajectory: dict[str, Any]) -> float:
    terminal = scalar_final_score(trajectory)
    soil = float(trajectory.get("mean_final_soil", 0.0))
    completion = float(trajectory.get("completion_rate", 0.0))
    return 0.6 * terminal + 0.3 * soil + 0.1 * completion


DEFAULT_GRADER = scalar_final_score
