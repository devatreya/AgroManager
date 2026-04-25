from __future__ import annotations

import asyncio

from pipeline import policy_rollout
from scripts.prepare_sft_data import (
    build_examples,
    quarter_chunks,
    select_train_trajectories,
    validate_example,
)
from tests.helpers import FakeHostedFarmSession


def _conversation() -> list[dict]:
    return [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Quarter 1 begins."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_farm_state", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "read_farm_state",
            "tool_call_id": "call_1",
            "content": "read_farm_state q=1 cash=150000.0 irrigation=False",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "commit_plan", "arguments": "{\"capital_action\": \"none\"}"},
                }
            ],
        },
        {
            "role": "tool",
            "name": "commit_plan",
            "tool_call_id": "call_2",
            "content": "commit_plan q=1 reward=0.1 cash=151000.0 capital=none",
        },
        {"role": "assistant", "content": "Quarter 1 plan committed."},
    ]


def test_policy_rollout_includes_tool_history(monkeypatch):
    class ShortSession(FakeHostedFarmSession):
        finish_after = 2

    monkeypatch.setattr(policy_rollout, "HostedFarmSession", ShortSession)
    rollout = asyncio.run(
        policy_rollout.run_policy_rollout(
            baseline_name="weather_aware_rotation",
            split="validation",
            task_id="task",
            openreward_env_id="env",
            capture_conversation=True,
        )
    )
    assert rollout["task_id"] == "task"
    assert rollout["split"] == "validation"
    assert rollout["completed"] is True
    assert rollout["terminal_cash"] == rollout["final_state"]["cash"]
    assert len(rollout["full_tool_interaction_history"]) == 10
    assert rollout["tool_calls"] == 10
    assert rollout["conversation"][0]["role"] == "system"


def test_select_train_trajectories_keeps_minimum():
    base = {
        "completed": True,
        "ever_bankrupt": False,
        "invalid_tool_calls": 0,
        "conversation": _conversation(),
        "full_tool_interaction_history": [{"tool_name": "read_farm_state"}],
        "baseline_name": "weather_aware_rotation",
        "split": "train",
        "terminal_cash": 200000.0,
    }
    trajectories = []
    for index in range(20):
        trajectories.append(
            {
                **base,
                "task_id": f"task-{index}",
                "terminal_score": float(index),
                "mean_final_soil": 0.5 + (index * 0.001),
            }
        )
    selected = select_train_trajectories(trajectories, top_quantile=0.25, min_keep=16)
    assert len(selected) == 16
    assert selected[0]["task_id"] == "task-19"


def test_build_examples_and_validate():
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for name in (
            "read_farm_state",
            "read_soil_report",
            "read_weather_history",
            "read_price_board",
            "commit_plan",
        )
    ]
    trajectory = {
        "task_id": "task-1",
        "split": "train",
        "baseline_name": "weather_aware_rotation",
        "conversation": _conversation(),
    }
    examples = build_examples([trajectory], quarters_per_example=4, tools=tools)
    assert len(examples) == 1
    validate_example(examples[0])
    assert quarter_chunks(_conversation(), 4)[0][0]["role"] == "system"


def test_validate_example_rejects_free_text_action():
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for name in (
            "read_farm_state",
            "read_soil_report",
            "read_weather_history",
            "read_price_board",
            "commit_plan",
        )
    ]
    example = {
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Quarter 1 begins."},
            {"role": "assistant", "content": "I will call read_farm_state now."},
        ],
        "tools": tools,
    }
    try:
        validate_example(example)
    except ValueError as exc:
        assert "Assistant free text" in str(exc)
    else:
        raise AssertionError("validate_example should reject assistant free-text action messages")
