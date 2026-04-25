from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_READ_TOOL_SEQUENCE, OPENREWARD_ENV_ID, project_root
from pipeline.farm_session import HostedFarmSession


SYNTHETIC_CONTINUATION_RE = re.compile(r"^Quarter\s+\d+\s+plan committed\.$")


def default_harvest_path(split: str) -> Path:
    return project_root() / "eval" / "trajectories" / split / "weather_aware_rotation.json"


def quarter_chunks(messages: list[dict[str, Any]], quarters_per_example: int) -> list[list[dict[str, Any]]]:
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Conversation must start with a system prompt.")

    system = messages[0]
    quarter_groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages[1:]:
        if message["role"] == "user" and current:
            quarter_groups.append(current)
            current = [message]
            continue
        current.append(message)
        if (
            message["role"] == "assistant"
            and not message.get("tool_calls")
            and SYNTHETIC_CONTINUATION_RE.match(str(message.get("content", "")))
        ):
            quarter_groups.append(current)
            current = []
    if current:
        quarter_groups.append(current)

    chunks = []
    for start in range(0, len(quarter_groups), quarters_per_example):
        payload = [system]
        for group in quarter_groups[start : start + quarters_per_example]:
            payload.extend(group)
        chunks.append(payload)
    return chunks


def load_harvest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run eval/run_baselines.py with "
            "--capture-conversation against the hosted environment first."
        )
    trajectories = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trajectories, list):
        raise ValueError(f"{path} does not contain a list of trajectories.")
    return trajectories


def trajectory_is_valid_teacher(trajectory: dict[str, Any]) -> bool:
    if not trajectory.get("completed"):
        return False
    if trajectory.get("ever_bankrupt"):
        return False
    if trajectory.get("invalid_tool_calls", 0):
        return False
    if not isinstance(trajectory.get("conversation"), list) or not trajectory["conversation"]:
        return False
    tool_history = trajectory.get("full_tool_interaction_history")
    if not isinstance(tool_history, list) or not tool_history:
        return False
    return True


def rank_trajectories(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trajectories,
        key=lambda trajectory: (
            1 if trajectory.get("completed") else 0,
            1 if not trajectory.get("ever_bankrupt") else 0,
            float(trajectory.get("terminal_score", 0.0)),
            float(trajectory.get("mean_final_soil", 0.0)),
            float(trajectory.get("terminal_cash", 0.0)),
        ),
        reverse=True,
    )


def select_train_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    top_quantile: float,
    min_keep: int,
) -> list[dict[str, Any]]:
    valid = [trajectory for trajectory in trajectories if trajectory_is_valid_teacher(trajectory)]
    ranked = rank_trajectories(valid)
    keep_count = max(min_keep, math.ceil(len(ranked) * top_quantile))
    keep_count = min(keep_count, len(ranked))
    return ranked[:keep_count]


def select_validation_trajectories(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [trajectory for trajectory in trajectories if trajectory_is_valid_teacher(trajectory)]
    return rank_trajectories(valid)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_examples(
    trajectories: list[dict[str, Any]],
    *,
    quarters_per_example: int,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples = []
    for trajectory in trajectories:
        for chunk in quarter_chunks(trajectory["conversation"], quarters_per_example):
            examples.append(
                {
                    "messages": chunk,
                    "tools": tools,
                    "task_id": trajectory["task_id"],
                    "split": trajectory["split"],
                    "baseline_name": trajectory["baseline_name"],
                }
            )
    return examples


def validate_tools(tools: list[dict[str, Any]]) -> None:
    allowed_tools = set(DEFAULT_READ_TOOL_SEQUENCE + ["commit_plan"])
    seen = set()
    if not isinstance(tools, list) or len(tools) != len(allowed_tools):
        raise ValueError("Tool specs must include exactly the five canonical tools.")
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError("Every tool spec must be a function tool.")
        function = tool.get("function", {})
        name = function.get("name")
        if name not in allowed_tools:
            raise ValueError(f"Unexpected tool name {name!r}.")
        if name in seen:
            raise ValueError(f"Duplicate tool name {name!r}.")
        seen.add(name)
        if not isinstance(function.get("parameters"), dict):
            raise ValueError(f"Tool {name!r} is missing parameters.")


def validate_example(example: dict[str, Any]) -> None:
    if not isinstance(example, dict):
        raise ValueError("Each example must be a JSON object.")

    messages = example.get("messages")
    tools = example.get("tools")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each example must contain non-empty messages.")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Each example must contain tools.")
    validate_tools(tools)

    if messages[0].get("role") != "system":
        raise ValueError("Each example must start with a system message.")

    allowed_tools = {tool["function"]["name"] for tool in tools}
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unexpected message role {role!r}.")
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                if len(tool_calls) != 1:
                    raise ValueError("Assistant messages must contain exactly one tool call.")
                tool_call = tool_calls[0]
                function = tool_call.get("function", {})
                name = function.get("name")
                if name not in allowed_tools:
                    raise ValueError(f"Unknown tool call {name!r}.")
                try:
                    json.loads(function.get("arguments", "{}"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Tool arguments for {name!r} are not valid JSON.") from exc
            else:
                content = str(message.get("content", "")).strip()
                if not SYNTHETIC_CONTINUATION_RE.match(content):
                    raise ValueError(
                        "Assistant free text is only allowed for the synthetic quarter continuation line."
                    )
        if role == "tool":
            name = message.get("name")
            if name not in allowed_tools:
                raise ValueError(f"Unknown tool result name {name!r}.")
            if not isinstance(message.get("content"), str):
                raise ValueError("Tool result content must be a string.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            validate_example(example)
            handle.write(json.dumps(example) + "\n")


def build_dataset_summary(
    *,
    train_selected: list[dict[str, Any]],
    validation_selected: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    top_quantile: float,
    quarters_per_example: int,
) -> dict[str, Any]:
    return {
        "top_quantile": top_quantile,
        "quarters_per_example": quarters_per_example,
        "selected_train_trajectories": len(train_selected),
        "selected_validation_trajectories": len(validation_selected),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "average_quarters_per_selected_train_trajectory": average(
            [float(trajectory.get("quarters_completed", 0)) for trajectory in train_selected]
        ),
        "average_quarters_per_selected_validation_trajectory": average(
            [float(trajectory.get("quarters_completed", 0)) for trajectory in validation_selected]
        ),
        "average_messages_per_train_example": average(
            [float(len(example["messages"])) for example in train_examples]
        ),
        "average_messages_per_validation_example": average(
            [float(len(example["messages"])) for example in validation_examples]
        ),
        "train_task_ids": [trajectory["task_id"] for trajectory in train_selected],
        "validation_task_ids": [trajectory["task_id"] for trajectory in validation_selected],
    }


async def main_async(args: argparse.Namespace) -> None:
    artifact_dir = project_root() / "artifacts" / "sft"
    train_harvest_path = Path(args.train_harvest_path) if args.train_harvest_path else default_harvest_path("train")
    validation_harvest_path = (
        Path(args.validation_harvest_path)
        if args.validation_harvest_path
        else default_harvest_path("validation")
    )

    train_harvest = load_harvest(train_harvest_path)
    validation_harvest = load_harvest(validation_harvest_path)

    train_selected = select_train_trajectories(
        train_harvest,
        top_quantile=args.top_quantile,
        min_keep=args.min_train_trajectories,
    )
    validation_selected = select_validation_trajectories(validation_harvest)
    tools = await HostedFarmSession.hosted_tool_specs(args.openreward_env_id)
    validate_tools(tools)

    train_examples = build_examples(
        train_selected,
        quarters_per_example=args.quarters_per_example,
        tools=tools,
    )
    validation_examples = build_examples(
        validation_selected,
        quarters_per_example=args.quarters_per_example,
        tools=tools,
    )

    write_json(artifact_dir / "selected_train_trajectories.json", train_selected)
    write_json(artifact_dir / "selected_validation_trajectories.json", validation_selected)
    write_jsonl(artifact_dir / "train.jsonl", train_examples)
    write_jsonl(artifact_dir / "validation.jsonl", validation_examples)

    summary = build_dataset_summary(
        train_selected=train_selected,
        validation_selected=validation_selected,
        train_examples=train_examples,
        validation_examples=validation_examples,
        top_quantile=args.top_quantile,
        quarters_per_example=args.quarters_per_example,
    )
    summary.update(
        {
            "train_harvest_path": str(train_harvest_path),
            "validation_harvest_path": str(validation_harvest_path),
        }
    )
    write_json(artifact_dir / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openreward-env-id", default=OPENREWARD_ENV_ID)
    parser.add_argument("--top-quantile", type=float, default=0.25)
    parser.add_argument("--quarters-per-example", type=int, default=4)
    parser.add_argument("--min-train-trajectories", type=int, default=16)
    parser.add_argument("--train-harvest-path")
    parser.add_argument("--validation-harvest-path")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
