from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines import BASELINES
from config import BASELINE_NAMES, OPENREWARD_ENV_ID, project_root
from grader import scalar_final_score
from pipeline.farm_session import HostedFarmSession
from pipeline.policy_rollout import run_policy_rollout


def summarize(trajectories: list[dict]) -> dict:
    count = len(trajectories) or 1
    return {
        "tasks": len(trajectories),
        "mean_terminal_cash": sum(t["final_state"]["cash"] for t in trajectories) / count,
        "mean_final_soil_health": sum(t["mean_final_soil"] for t in trajectories) / count,
        "bankruptcy_rate": sum(1 for t in trajectories if t["ever_bankrupt"]) / count,
        "completion_rate": sum(t["completion_rate"] for t in trajectories) / count,
        "mean_total_episode_reward": sum(t["total_reward"] for t in trajectories) / count,
        "mean_terminal_score": sum(t["terminal_score"] for t in trajectories) / count,
        "mean_grader_score": sum(scalar_final_score(t) for t in trajectories) / count,
    }


def per_task_metrics(trajectories: list[dict]) -> list[dict]:
    metrics = []
    for trajectory in trajectories:
        metrics.append(
            {
                "task_id": trajectory["task_id"],
                "split": trajectory["split"],
                "completed": trajectory["completed"],
                "bankruptcy_flag": trajectory["ever_bankrupt"],
                "quarters_completed": trajectory["quarters_completed"],
                "completion_rate": trajectory["completion_rate"],
                "terminal_score": trajectory["terminal_score"],
                "total_reward": trajectory["total_reward"],
                "mean_final_soil": trajectory["mean_final_soil"],
                "terminal_cash": trajectory["terminal_cash"],
                "tool_calls": trajectory["tool_calls"],
                "invalid_tool_calls": trajectory["invalid_tool_calls"],
            }
        )
    return metrics


async def main_async(args: argparse.Namespace) -> None:
    tasks = await HostedFarmSession.list_tasks(args.split, openreward_env_id=args.openreward_env_id)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    baseline_names = args.baselines or list(BASELINE_NAMES)

    results_dir = project_root() / "eval" / "results"
    trajectories_dir = project_root() / "eval" / "trajectories" / args.split
    results_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    payload = {"split": args.split, "openreward_env_id": args.openreward_env_id, "baselines": {}}

    for baseline_name in baseline_names:
        if baseline_name not in BASELINES:
            raise SystemExit(f"Unknown baseline {baseline_name!r}")
        rollouts = []
        for index, task in enumerate(tasks, start=1):
            print(
                f"[{baseline_name}] split={args.split} task={index}/{len(tasks)} "
                f"task_id={task.task_spec['task_id']}",
                flush=True,
            )
            rollouts.append(
                await run_policy_rollout(
                    baseline_name=baseline_name,
                    split=args.split,
                    task_id=task.task_spec["task_id"],
                    openreward_env_id=args.openreward_env_id,
                    capture_conversation=args.capture_conversation,
                )
            )
        per_task = per_task_metrics(rollouts)
        payload["baselines"][baseline_name] = {
            "summary": summarize(rollouts),
            "per_task_metrics": per_task,
            "per_task_metrics_path": str(trajectories_dir / f"{baseline_name}.metrics.json"),
            "trajectories_path": str(trajectories_dir / f"{baseline_name}.json"),
        }
        (trajectories_dir / f"{baseline_name}.metrics.json").write_text(
            json.dumps(per_task, indent=2),
            encoding="utf-8",
        )
        (trajectories_dir / f"{baseline_name}.json").write_text(
            json.dumps(rollouts, indent=2),
            encoding="utf-8",
        )

    (results_dir / f"{args.split}_baselines.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--openreward-env-id", default=OPENREWARD_ENV_ID)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--baseline", dest="baselines", action="append")
    parser.add_argument("--capture-conversation", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
