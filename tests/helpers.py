from __future__ import annotations

import copy
from types import SimpleNamespace

from pipeline.farm_session import HostedToolResult


class FakeHostedFarmSession:
    finish_after = 40
    raise_on_commit: int | None = None

    def __init__(self, split: str, task_id: str, openreward_env_id: str):
        self.split = split
        self.task_id = task_id
        self.openreward_env_id = openreward_env_id
        self.task = SimpleNamespace(task_spec={"task_id": task_id, "split": split})
        self.prompt = "Hosted test prompt"
        self.last_state = {
            "quarter": 1,
            "cash": 150000.0,
            "irrigation_owned": False,
            "finished": False,
            "plots": [
                {"plot_id": i, "crop": "wheat", "previous_crop": "barley", "soil_health": 0.8}
                for i in range(4)
            ],
        }
        self.last_episode_metrics = {
            "terminal_score": 0.0,
            "ending_cash": 150000.0,
            "mean_final_soil": 0.8,
            "ever_bankrupt": False,
            "quarters_completed": 0,
            "completion_rate": 0.0,
            "finished": False,
            "completed_all_quarters": False,
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get_prompt(self) -> str:
        return self.prompt

    async def get_tool_specs(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}
            for name in (
                "read_farm_state",
                "read_soil_report",
                "read_weather_history",
                "read_price_board",
                "commit_plan",
            )
        ]

    async def call_tool(self, tool_name: str, payload: dict | None = None) -> HostedToolResult:
        payload = payload or {}
        if tool_name == "read_farm_state":
            return HostedToolResult(
                tool_name=tool_name,
                text="farm state",
                metadata={"tool": tool_name, "state": copy.deepcopy(self.last_state)},
                reward=None,
                finished=False,
                state=copy.deepcopy(self.last_state),
                episode_metrics=copy.deepcopy(self.last_episode_metrics),
            )
        if tool_name == "read_soil_report":
            report = [
                {
                    "plot_id": i,
                    "soil_health": 0.8,
                    "organic_matter": 0.8,
                    "structure": 0.8,
                    "ph": 0.9,
                    "nutrient_balance": 0.8,
                }
                for i in range(4)
            ]
            return HostedToolResult(
                tool_name=tool_name,
                text="soil report",
                metadata={"tool": tool_name, "state": copy.deepcopy(self.last_state), "report": report},
                reward=None,
                finished=False,
                state=copy.deepcopy(self.last_state),
                episode_metrics=copy.deepcopy(self.last_episode_metrics),
            )
        if tool_name == "read_weather_history":
            history = [
                {"year": 2025, "quarter": 1, "regime": "normal", "rainfall_index": 1.0, "temperature_index": 1.0},
                {"year": 2025, "quarter": 2, "regime": "dry", "rainfall_index": 0.7, "temperature_index": 1.1},
            ]
            return HostedToolResult(
                tool_name=tool_name,
                text="weather history",
                metadata={"tool": tool_name, "state": copy.deepcopy(self.last_state), "history": history},
                reward=None,
                finished=False,
                state=copy.deepcopy(self.last_state),
                episode_metrics=copy.deepcopy(self.last_episode_metrics),
            )
        if tool_name == "read_price_board":
            board = {
                "crop_prices_gbp_per_acre": {
                    "wheat": 700.0,
                    "barley": 620.0,
                    "oilseed_rape": 760.0,
                    "field_beans": 540.0,
                    "cover_crop": 0.0,
                    "fallow": 0.0,
                },
                "fertiliser_costs_gbp_per_acre": {"low": 20.0, "medium": 45.0, "high": 75.0},
                "irrigation_cost_gbp": 35000.0,
            }
            return HostedToolResult(
                tool_name=tool_name,
                text="price board",
                metadata={"tool": tool_name, "state": copy.deepcopy(self.last_state), "price_board": board},
                reward=None,
                finished=False,
                state=copy.deepcopy(self.last_state),
                episode_metrics=copy.deepcopy(self.last_episode_metrics),
            )
        if self.raise_on_commit and self.last_state["quarter"] == self.raise_on_commit:
            raise RuntimeError("forced tool failure")

        committed_quarter = self.last_state["quarter"]
        finished = committed_quarter >= self.finish_after
        self.last_state["cash"] += 1000.0
        self.last_state["quarter"] = min(committed_quarter + 1, self.finish_after)
        self.last_state["finished"] = finished
        self.last_episode_metrics = {
            "terminal_score": 0.9 if finished else 0.0,
            "ending_cash": self.last_state["cash"],
            "mean_final_soil": 0.82,
            "ever_bankrupt": False,
            "quarters_completed": committed_quarter,
            "completion_rate": committed_quarter / 40.0,
            "finished": finished,
            "completed_all_quarters": finished,
        }
        return HostedToolResult(
            tool_name=tool_name,
            text=f"Quarter {committed_quarter} committed.",
            metadata={
                "tool": tool_name,
                "state": copy.deepcopy(self.last_state),
                "step": committed_quarter,
                "episode_metrics": copy.deepcopy(self.last_episode_metrics),
                "action": payload,
                "result": {"pnl": 1000.0, "terminal_score": self.last_episode_metrics["terminal_score"]},
            },
            reward=0.1,
            finished=finished,
            state=copy.deepcopy(self.last_state),
            episode_metrics=copy.deepcopy(self.last_episode_metrics),
        )


class SequenceInferencer:
    def __init__(self, sequence: list[str] | None = None):
        self.sequence = sequence or (
            ["read_farm_state", "read_soil_report", "read_weather_history", "read_price_board", "commit_plan"] * 60
        )
        self.index = 0

    async def __call__(self, *, messages, tools, temperature, max_completion_tokens):
        del messages, tools, temperature, max_completion_tokens
        tool_name = self.sequence[self.index]
        self.index += 1
        assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{self.index}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        }
        from pipeline.art_rollout import ToolInferenceResponse

        return ToolInferenceResponse(
            tool_name=tool_name,
            arguments={},
            call_id=f"call_{self.index}",
            assistant_message=assistant_message,
            raw_choice=None,
            invalid_json=False,
        )


class MissingToolInferencer:
    async def __call__(self, *, messages, tools, temperature, max_completion_tokens):
        del messages, tools, temperature, max_completion_tokens
        from pipeline.art_rollout import ToolInferenceResponse

        return ToolInferenceResponse(
            tool_name=None,
            arguments=None,
            call_id=None,
            assistant_message={"role": "assistant", "content": "no tool"},
        )
