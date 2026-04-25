from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import OPENREWARD_ENV_ID, project_root
from pipeline.farm_session import HostedFarmSession
from pipeline.policy_rollout import run_policy_rollout


def quarter_chunks(messages: list[dict], quarters_per_example: int) -> list[list[dict]]:
    system = messages[0]
    quarter_groups: list[list[dict]] = []
    current: list[dict] = []
    for message in messages[1:]:
        if message["role"] == "user" and current:
            quarter_groups.append(current)
            current = [message]
            continue
        current.append(message)
        if message["role"] == "assistant" and str(message.get("content", "")).endswith("plan committed."):
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


async def build_split(
    split: str,
    output_path: Path,
    *,
    openreward_env_id: str,
    top_quantile: float,
    quarters_per_example: int,
) -> None:
    tasks = await HostedFarmSession.list_tasks(split, openreward_env_id=openreward_env_id)
    scored = []
    for task in tasks:
        rollout = await run_policy_rollout(
            baseline_name="weather_aware_rotation",
            split=split,
            task_id=task.task_spec["task_id"],
            openreward_env_id=openreward_env_id,
            capture_conversation=False,
        )
        scored.append((rollout["terminal_score"], task.task_spec["task_id"]))

    scored.sort(reverse=True)
    keep_count = max(1, math.ceil(len(scored) * top_quantile))
    selected_task_ids = [task_id for _, task_id in scored[:keep_count]]
    tools = await HostedFarmSession.hosted_tool_specs(openreward_env_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task_id in selected_task_ids:
            rollout = await run_policy_rollout(
                baseline_name="weather_aware_rotation",
                split=split,
                task_id=task_id,
                openreward_env_id=openreward_env_id,
                capture_conversation=True,
            )
            for chunk in quarter_chunks(rollout["conversation"], quarters_per_example):
                handle.write(json.dumps({"messages": chunk, "tools": tools}) + "\n")
    print(f"Wrote {output_path}")


async def main_async(args: argparse.Namespace) -> None:
    artifact_dir = project_root() / "artifacts" / "sft"
    await build_split(
        "train",
        artifact_dir / "train.jsonl",
        openreward_env_id=args.openreward_env_id,
        top_quantile=args.top_quantile,
        quarters_per_example=args.quarters_per_example,
    )
    await build_split(
        "validation",
        artifact_dir / "validation.jsonl",
        openreward_env_id=args.openreward_env_id,
        top_quantile=args.top_quantile,
        quarters_per_example=args.quarters_per_example,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openreward-env-id", default=OPENREWARD_ENV_ID)
    parser.add_argument("--top-quantile", type=float, default=0.25)
    parser.add_argument("--quarters-per-example", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
