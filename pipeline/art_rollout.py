from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Protocol

from config import COMMIT_TOOL_NAME, DEFAULT_MAX_TOOL_CALLS, DEFAULT_READ_TOOL_SEQUENCE

from .farm_session import HostedFarmSession
from .tool_transcript import CompactToolTranscript


@dataclass
class ToolInferenceResponse:
    tool_name: str | None
    arguments: dict[str, Any] | None
    call_id: str | None
    assistant_message: dict[str, Any]
    raw_choice: Any = None
    invalid_json: bool = False


class ToolInferencer(Protocol):
    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_completion_tokens: int,
    ) -> ToolInferenceResponse: ...


def _assistant_message_from_choice(choice: Any) -> dict[str, Any]:
    tool_calls = [
        tool_call.model_dump(mode="json")
        for tool_call in (choice.message.tool_calls or [])
    ]
    return {
        "role": "assistant",
        "content": choice.message.content or "",
        "tool_calls": tool_calls,
    }


class OpenAIToolInferencer:
    def __init__(self, client: Any, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_completion_tokens: int,
    ) -> ToolInferenceResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls or []
        if len(tool_calls) != 1:
            return ToolInferenceResponse(
                tool_name=None,
                arguments=None,
                call_id=None,
                assistant_message=_assistant_message_from_choice(choice),
                raw_choice=choice,
            )
        tool_call = tool_calls[0]
        invalid_json = False
        arguments = None
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            invalid_json = True
        return ToolInferenceResponse(
            tool_name=tool_call.function.name,
            arguments=arguments,
            call_id=tool_call.id,
            assistant_message=_assistant_message_from_choice(choice),
            raw_choice=choice,
            invalid_json=invalid_json,
        )


def _tool_message(tool_name: str, call_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": tool_name,
        "tool_call_id": call_id,
        "content": content,
    }


def _termination_summary(
    *,
    trajectory: Any,
    metrics: dict[str, Any],
    final_state: dict[str, Any] | None,
    final_episode_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    final_state = final_state or {}
    final_episode_metrics = final_episode_metrics or {}
    metrics.update(
        {
            "terminal_score": float(final_episode_metrics.get("terminal_score", 0.0)),
            "ending_cash": float(final_episode_metrics.get("ending_cash", final_state.get("cash", 0.0) or 0.0)),
            "mean_final_soil": float(final_episode_metrics.get("mean_final_soil", 0.0)),
            "ever_bankrupt": bool(final_episode_metrics.get("ever_bankrupt", False)),
            "quarters_completed": int(final_episode_metrics.get("quarters_completed", 0)),
            "completion_rate": float(final_episode_metrics.get("completion_rate", 0.0)),
            "finished": bool(final_episode_metrics.get("finished", False)),
            "completed_all_quarters": bool(final_episode_metrics.get("completed_all_quarters", False)),
            "final_state": final_state,
            "trajectory": trajectory,
        }
    )
    return metrics


async def rollout_with_inferencer(
    *,
    inferencer: ToolInferencer,
    split: str,
    task_id: str,
    openreward_env_id: str,
    temperature: float = 0.0,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_completion_tokens: int = 256,
) -> dict[str, Any]:
    try:
        import art
    except ImportError:
        art = None

    transcript = CompactToolTranscript()
    commit_count = 0
    tool_call_count = 0
    invalid_tool_calls = 0
    total_reward = 0.0
    total_pnl = 0.0
    last_tool_name: str | None = None
    current_quarter = 1
    next_tool_hint = DEFAULT_READ_TOOL_SEQUENCE[0]
    system_prompt = ""

    if art is not None:
        trajectory = art.Trajectory(messages_and_choices=[], reward=0.0, metrics={})
    else:
        trajectory = {"messages_and_choices": []}

    try:
        async with HostedFarmSession(
            split=split,
            task_id=task_id,
            openreward_env_id=openreward_env_id,
        ) as session:
            system_prompt = await session.get_prompt()
            system_message = {"role": "system", "content": system_prompt}
            if art is not None:
                trajectory.messages_and_choices.append(system_message)
            else:
                trajectory["messages_and_choices"].append(system_message)

            tools = await session.get_tool_specs()
            termination_reason = "finished"

            while True:
                if tool_call_count >= max_tool_calls:
                    termination_reason = "tool_budget_exhausted"
                    break

                user_message = {
                    "role": "user",
                    "content": transcript.build_user_prompt(next_tool_hint=next_tool_hint),
                }
                messages = [system_message, user_message]
                response = await inferencer(
                    messages=messages,
                    tools=copy.deepcopy(tools),
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                )

                if response.invalid_json and temperature > 0.0:
                    response = await inferencer(
                        messages=messages,
                        tools=copy.deepcopy(tools),
                        temperature=0.0,
                        max_completion_tokens=max_completion_tokens,
                    )

                if art is not None:
                    trajectory.messages_and_choices.append(user_message)
                    trajectory.messages_and_choices.append(response.raw_choice or response.assistant_message)
                else:
                    trajectory["messages_and_choices"].append(user_message)
                    trajectory["messages_and_choices"].append(response.assistant_message)

                if response.tool_name is None or response.arguments is None:
                    invalid_tool_calls += 1
                    termination_reason = "missing_tool_call"
                    break

                tool_call_count += 1
                last_tool_name = response.tool_name
                if response.tool_name == COMMIT_TOOL_NAME:
                    commit_count += 1

                call_id = response.call_id or f"{task_id}_{tool_call_count}_{response.tool_name}"
                current_quarter = int((session.last_state or {}).get("quarter", current_quarter))
                result = await session.call_tool(response.tool_name, response.arguments)
                summary = transcript.record(result)
                total_reward += float(result.reward or 0.0)
                if response.tool_name == COMMIT_TOOL_NAME:
                    total_pnl += float(result.metadata.get("result", {}).get("pnl", 0.0))

                tool_message = _tool_message(response.tool_name, call_id, summary)
                if art is not None:
                    trajectory.messages_and_choices.append(tool_message)
                else:
                    trajectory["messages_and_choices"].append(tool_message)

                if result.finished:
                    termination_reason = "finished"
                    break

                if response.tool_name in DEFAULT_READ_TOOL_SEQUENCE:
                    next_index = DEFAULT_READ_TOOL_SEQUENCE.index(response.tool_name) + 1
                    next_tool_hint = (
                        DEFAULT_READ_TOOL_SEQUENCE[next_index]
                        if next_index < len(DEFAULT_READ_TOOL_SEQUENCE)
                        else COMMIT_TOOL_NAME
                    )
                else:
                    next_tool_hint = DEFAULT_READ_TOOL_SEQUENCE[0]

            metrics = {
                "task_id": task_id,
                "split": split,
                "tool_calls": tool_call_count,
                "quarter_commits": commit_count,
                "commit_attempts": commit_count,
                "invalid_tool_calls": invalid_tool_calls,
                "total_reward": total_reward,
                "total_pnl": total_pnl,
                "termination_reason": termination_reason,
                "last_tool_name": last_tool_name,
            }
            return _termination_summary(
                trajectory=trajectory,
                metrics=metrics,
                final_state=session.last_state,
                final_episode_metrics=session.last_episode_metrics,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Rollout exception for task_id={task_id} split={split} "
            f"current_quarter={current_quarter} commit_count={commit_count} "
            f"tool_call_count={tool_call_count} last_tool_name={last_tool_name}"
        ) from exc


async def rollout_model_on_task(
    *,
    model: Any,
    split: str,
    task_id: str,
    openreward_env_id: str,
    temperature: float = 0.0,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_completion_tokens: int = 256,
) -> dict[str, Any]:
    client = model.openai_client()
    inferencer = OpenAIToolInferencer(client=client, model_name=model.get_inference_name())
    return await rollout_with_inferencer(
        inferencer=inferencer,
        split=split,
        task_id=task_id,
        openreward_env_id=openreward_env_id,
        temperature=temperature,
        max_tool_calls=max_tool_calls,
        max_completion_tokens=max_completion_tokens,
    )
