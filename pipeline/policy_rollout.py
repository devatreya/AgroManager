from __future__ import annotations

import copy
import json
from typing import Any

from baselines import BASELINES
from config import (
    COMMIT_TOOL_NAME,
    DEFAULT_READ_TOOL_SEQUENCE,
    OPENREWARD_ENV_ID,
)

from .farm_session import HostedFarmSession
from .tool_transcript import CompactToolTranscript


def _assistant_tool_call(tool_name: str, payload: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(payload),
                },
            }
        ],
    }


def _tool_result_message(tool_name: str, call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": tool_name,
        "tool_call_id": call_id,
        "content": content,
    }


async def run_policy_rollout(
    *,
    baseline_name: str,
    split: str,
    task_id: str,
    openreward_env_id: str = OPENREWARD_ENV_ID,
    capture_conversation: bool = False,
) -> dict[str, Any]:
    if baseline_name not in BASELINES:
        raise KeyError(f"Unknown baseline {baseline_name!r}")
    policy = BASELINES[baseline_name]

    async with HostedFarmSession(
        split=split,
        task_id=task_id,
        openreward_env_id=openreward_env_id,
    ) as session:
        transcript = CompactToolTranscript()
        trajectory = {
            "task_id": task_id,
            "split": split,
            "task_spec": dict(session.task.task_spec),
            "steps": [],
            "final_state": None,
            "terminal_score": 0.0,
            "terminal_cash": 0.0,
            "completed": False,
            "ever_bankrupt": False,
            "quarters_completed": 0,
            "total_reward": 0.0,
            "total_pnl": 0.0,
            "mean_final_soil": 0.0,
            "tool_calls": 0,
            "invalid_tool_calls": 0,
            "full_tool_interaction_history": [],
            "baseline_name": baseline_name,
        }
        conversation: list[dict[str, Any]] = []
        if capture_conversation:
            conversation.append({"role": "system", "content": await session.get_prompt()})
            conversation.append(
                {
                    "role": "user",
                    "content": "Quarter 1 begins. Use the required five-tool sequence and only tool calls.",
                }
            )

        quarter_number = 1
        finished = False
        while not finished:
            quarter_step: dict[str, Any] = {"quarter": quarter_number, "reads": {}}
            read_results = {}
            for tool_name in DEFAULT_READ_TOOL_SEQUENCE:
                payload = {}
                if tool_name == "read_soil_report":
                    payload = {"plots": [0, 1, 2, 3]}
                elif tool_name == "read_weather_history":
                    payload = {"lookback_quarters": 4}
                result = await session.call_tool(tool_name, payload)
                summary = transcript.record(result)
                read_results[tool_name] = result
                quarter_step["reads"][tool_name] = summary
                trajectory["tool_calls"] += 1
                trajectory["full_tool_interaction_history"].append(
                    {
                        "quarter": quarter_number,
                        "tool_name": tool_name,
                        "payload": copy.deepcopy(payload),
                        "summary": summary,
                        "text": result.text,
                        "reward": result.reward,
                        "finished": result.finished,
                        "metadata": copy.deepcopy(result.metadata),
                        "state": copy.deepcopy(result.state),
                        "episode_metrics": copy.deepcopy(result.episode_metrics),
                    }
                )

                if capture_conversation:
                    call_id = f"{task_id}_{quarter_number}_{tool_name}"
                    conversation.append(_assistant_tool_call(tool_name, payload, call_id))
                    conversation.append(_tool_result_message(tool_name, call_id, summary))

            state = read_results["read_farm_state"].state or {}
            soil_report = read_results["read_soil_report"].metadata.get("report", [])
            weather_history = read_results["read_weather_history"].metadata.get("history", [])
            price_board = read_results["read_price_board"].metadata.get("price_board", {})
            commit_payload = policy(state, soil_report, weather_history, price_board)

            commit_result = await session.call_tool(COMMIT_TOOL_NAME, commit_payload)
            commit_summary = transcript.record(commit_result)
            result_payload = commit_result.metadata.get("result", {})

            quarter_step["action"] = commit_payload
            quarter_step["commit"] = commit_summary
            quarter_step["reward"] = commit_result.reward or 0.0
            quarter_step["pnl"] = result_payload.get("pnl", 0.0)
            quarter_step["terminal_score"] = result_payload.get("terminal_score", 0.0)
            quarter_step["finished"] = commit_result.finished
            trajectory["steps"].append(quarter_step)
            trajectory["tool_calls"] += 1
            trajectory["full_tool_interaction_history"].append(
                {
                    "quarter": quarter_number,
                    "tool_name": COMMIT_TOOL_NAME,
                    "payload": copy.deepcopy(commit_payload),
                    "summary": commit_summary,
                    "text": commit_result.text,
                    "reward": commit_result.reward,
                    "finished": commit_result.finished,
                    "metadata": copy.deepcopy(commit_result.metadata),
                    "state": copy.deepcopy(commit_result.state),
                    "episode_metrics": copy.deepcopy(commit_result.episode_metrics),
                }
            )

            if capture_conversation:
                call_id = f"{task_id}_{quarter_number}_{COMMIT_TOOL_NAME}"
                conversation.append(_assistant_tool_call(COMMIT_TOOL_NAME, commit_payload, call_id))
                conversation.append(_tool_result_message(COMMIT_TOOL_NAME, call_id, commit_summary))
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"Quarter {quarter_number} plan committed.",
                    }
                )

            trajectory["total_reward"] += float(commit_result.reward or 0.0)
            trajectory["total_pnl"] += float(result_payload.get("pnl", 0.0))

            state = commit_result.state or {}
            episode_metrics = commit_result.episode_metrics or {}
            finished = bool(commit_result.finished)
            quarter_number += 1

            if not finished and capture_conversation:
                conversation.append(
                    {
                        "role": "user",
                        "content": f"Quarter {state.get('quarter')} begins. Continue with the required five-tool sequence and only tool calls.",
                    }
                )

        final_state = session.last_state or {}
        final_metrics = session.last_episode_metrics or {}
        trajectory["final_state"] = final_state
        trajectory["terminal_score"] = float(final_metrics.get("terminal_score", 0.0))
        trajectory["terminal_cash"] = float(
            final_metrics.get("ending_cash", final_state.get("cash", 0.0) or 0.0)
        )
        trajectory["completed"] = bool(final_metrics.get("completed_all_quarters", False))
        trajectory["ever_bankrupt"] = bool(final_metrics.get("ever_bankrupt", False))
        trajectory["quarters_completed"] = int(final_metrics.get("quarters_completed", 0))
        trajectory["mean_final_soil"] = float(final_metrics.get("mean_final_soil", 0.0))
        trajectory["completion_rate"] = float(final_metrics.get("completion_rate", 0.0))
        if capture_conversation:
            trajectory["conversation"] = conversation
            trajectory["system_prompt"] = await session.get_prompt()
        return trajectory
