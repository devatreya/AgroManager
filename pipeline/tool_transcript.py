from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .farm_session import HostedToolResult


def _plot_bits(plot: dict[str, Any]) -> str:
    return (
        f"P{plot['plot_id'] + 1}:{plot['crop']}"
        f"/prev={plot['previous_crop']}"
        f"/soil={plot['soil_health']:.3f}"
    )


@dataclass
class CompactToolTranscript:
    prior_commit_lines: list[str] = field(default_factory=list)
    current_quarter_lines: list[str] = field(default_factory=list)

    def reset_current_quarter(self) -> None:
        self.current_quarter_lines = []

    def record(self, result: HostedToolResult) -> str:
        summary = self._summarize(result)
        if result.tool_name == "commit_plan":
            self.prior_commit_lines.append(summary)
            self.prior_commit_lines = self.prior_commit_lines[-8:]
            self.reset_current_quarter()
        else:
            self.current_quarter_lines.append(summary)
        return summary

    def build_user_prompt(self, next_tool_hint: str | None = None) -> str:
        lines = []
        if self.prior_commit_lines:
            lines.append("Recent committed progress:")
            lines.extend(self.prior_commit_lines[-4:])
        if self.current_quarter_lines:
            lines.append("Current quarter observations:")
            lines.extend(self.current_quarter_lines[-8:])
        if next_tool_hint:
            lines.append(f"Call exactly one tool now. Expected next tool: {next_tool_hint}.")
        else:
            lines.append("Call exactly one tool now.")
        return "\n".join(lines)

    def _summarize(self, result: HostedToolResult) -> str:
        state = result.state or {}
        if result.tool_name == "read_farm_state":
            plots = ", ".join(_plot_bits(plot) for plot in state.get("plots", []))
            return (
                f"read_farm_state q={state.get('quarter')} cash={state.get('cash')} "
                f"irrigation={state.get('irrigation_owned')} {plots}"
            )
        if result.tool_name == "read_soil_report":
            report = result.metadata.get("report", [])
            bits = [
                f"P{row['plot_id'] + 1}:OM={row['organic_matter']:.3f}/structure={row['structure']:.3f}/pH={row['ph']:.3f}/nutrients={row['nutrient_balance']:.3f}/soil={row['soil_health']:.3f}"
                for row in report
            ]
            return "read_soil_report " + ", ".join(bits)
        if result.tool_name == "read_weather_history":
            history = result.metadata.get("history", [])
            bits = [
                f"{row['year']}-Q{row['quarter']}:{row['regime']}/rain={row['rainfall_index']:.2f}/temp={row['temperature_index']:.2f}"
                for row in history[-6:]
            ]
            return "read_weather_history " + ", ".join(bits)
        if result.tool_name == "read_price_board":
            board = result.metadata.get("price_board", {})
            crop_bits = [
                f"{crop}={price:.0f}"
                for crop, price in board.get("crop_prices_gbp_per_acre", {}).items()
            ]
            fert_bits = [
                f"{level}={price:.0f}"
                for level, price in board.get("fertiliser_costs_gbp_per_acre", {}).items()
            ]
            irrigation_cost = board.get("irrigation_cost_gbp")
            irrigation_line = (
                f" irrigation={irrigation_cost:.0f}" if irrigation_cost is not None else ""
            )
            return (
                "read_price_board "
                + " ".join(crop_bits)
                + " fert="
                + ",".join(fert_bits)
                + irrigation_line
            )
        action = result.metadata.get("action", {})
        plots = []
        for name in ("plot_1", "plot_2", "plot_3", "plot_4"):
            if name in action:
                plan = action[name]
                plots.append(
                    f"{name}={plan['crop']}/{plan['fertiliser']}/{plan['pest_control']}"
                )
        terminal_line = ""
        if result.finished and result.episode_metrics:
            terminal_line = (
                f" terminal_score={result.episode_metrics.get('terminal_score')}"
            )
        return (
            f"commit_plan q={result.metadata.get('step')} reward={result.reward} "
            f"cash={state.get('cash')} capital={action.get('capital_action')} "
            + " ".join(plots)
            + terminal_line
        )
