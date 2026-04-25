from __future__ import annotations

import asyncio

import pytest

from pipeline import art_rollout, policy_rollout
from tests.helpers import FakeHostedFarmSession, MissingToolInferencer, SequenceInferencer


def test_rollout_finishes_with_default_budget(monkeypatch):
    monkeypatch.setattr(art_rollout, "HostedFarmSession", FakeHostedFarmSession)
    result = asyncio.run(
        art_rollout.rollout_with_inferencer(
            inferencer=SequenceInferencer(),
            split="validation",
            task_id="task",
            openreward_env_id="env",
            max_tool_calls=240,
        )
    )
    assert result["termination_reason"] == "finished"
    assert result["quarter_commits"] == 40


def test_rollout_exhausts_at_160(monkeypatch):
    monkeypatch.setattr(art_rollout, "HostedFarmSession", FakeHostedFarmSession)
    result = asyncio.run(
        art_rollout.rollout_with_inferencer(
            inferencer=SequenceInferencer(),
            split="validation",
            task_id="task",
            openreward_env_id="env",
            max_tool_calls=160,
        )
    )
    assert result["termination_reason"] == "tool_budget_exhausted"
    assert result["quarter_commits"] == 32


def test_rollout_missing_tool_call(monkeypatch):
    monkeypatch.setattr(art_rollout, "HostedFarmSession", FakeHostedFarmSession)
    result = asyncio.run(
        art_rollout.rollout_with_inferencer(
            inferencer=MissingToolInferencer(),
            split="validation",
            task_id="task",
            openreward_env_id="env",
        )
    )
    assert result["termination_reason"] == "missing_tool_call"


def test_rollout_exception_context(monkeypatch):
    class ExplodingSession(FakeHostedFarmSession):
        raise_on_commit = 2

    monkeypatch.setattr(art_rollout, "HostedFarmSession", ExplodingSession)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            art_rollout.rollout_with_inferencer(
                inferencer=SequenceInferencer(),
                split="validation",
                task_id="task",
                openreward_env_id="env",
            )
        )
    message = str(exc_info.value)
    assert "task_id=task" in message
    assert "split=validation" in message
    assert "current_quarter=2" in message


def test_policy_rollout_captures_terminal_state(monkeypatch):
    class ShortSession(FakeHostedFarmSession):
        finish_after = 2

    monkeypatch.setattr(policy_rollout, "HostedFarmSession", ShortSession)
    rollout = asyncio.run(
        policy_rollout.run_policy_rollout(
            baseline_name="weather_aware_rotation",
            split="validation",
            task_id="task",
            openreward_env_id="env",
        )
    )
    assert rollout["final_state"] is not None
    assert rollout["terminal_score"] == 0.9
