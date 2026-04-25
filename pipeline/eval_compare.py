from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import EVAL_DEFAULTS, PIPELINE_META, comparison_path
from pipeline.modal_common import build_app, dump_json

app, image, volumes, secrets = build_app("eval-compare")


def summarize_policy_rollouts(rollouts: list[dict]) -> dict[str, float]:
    count = max(1, len(rollouts))
    return {
        "mean_terminal_cash": sum(r["terminal_cash"] for r in rollouts) / count,
        "mean_final_soil_health": sum(r["mean_final_soil"] for r in rollouts) / count,
        "bankruptcy_rate": sum(1 for r in rollouts if r["ever_bankrupt"]) / count,
        "completion_rate": sum(r["completion_rate"] for r in rollouts) / count,
        "mean_total_episode_reward": sum(r["total_reward"] for r in rollouts) / count,
        "mean_terminal_score": sum(r["terminal_score"] for r in rollouts) / count,
    }


def summarize_model_rollouts(rollouts: list[dict]) -> dict[str, object]:
    count = max(1, len(rollouts))
    termination_counts = Counter(r["termination_reason"] for r in rollouts)
    return {
        "mean_terminal_cash": sum(r["ending_cash"] for r in rollouts) / count,
        "mean_final_soil_health": sum(r["mean_final_soil"] for r in rollouts) / count,
        "bankruptcy_rate": sum(1 for r in rollouts if r["ever_bankrupt"]) / count,
        "completion_rate": sum(r["completion_rate"] for r in rollouts) / count,
        "completed_all_quarters_rate": sum(
            1 for r in rollouts if r["completed_all_quarters"]
        )
        / count,
        "mean_total_episode_reward": sum(r["total_reward"] for r in rollouts) / count,
        "mean_terminal_score": sum(r["terminal_score"] for r in rollouts) / count,
        "invalid_tool_call_count": sum(r["invalid_tool_calls"] for r in rollouts),
        "missing_tool_call_count": termination_counts["missing_tool_call"],
        "tool_budget_exhausted_count": termination_counts["tool_budget_exhausted"],
        "finished_count": termination_counts["finished"],
        "exception_count": termination_counts["exception"],
        "termination_reasons": dict(termination_counts),
    }


@app.function(
    gpu="H100:2",
    timeout=60 * 60 * 6,
    volumes=volumes,
    secrets=secrets,
)
async def run_eval_compare(
    split: str,
    max_tasks: int,
    openreward_env_id: str,
    temperature: float,
    max_tool_calls: int,
    max_completion_tokens: int,
) -> dict:
    import art
    from art.local import LocalBackend

    from config import MODEL_NAME, MODAL_VOLUME_MOUNTS, PROJECT_NAME, QWEN_MODEL_NAME
    from pipeline.art_rollout import rollout_model_on_task
    from pipeline.config import art_internal_model_config
    from pipeline.farm_session import HostedFarmSession
    from pipeline.policy_rollout import run_policy_rollout

    backend = LocalBackend(path=MODAL_VOLUME_MOUNTS["art"])
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=QWEN_MODEL_NAME,
        base_path=MODAL_VOLUME_MOUNTS["art"],
        _internal_config=art_internal_model_config(mode="shared"),
    )
    await model.register(backend)

    tasks = await HostedFarmSession.list_tasks(split, openreward_env_id=openreward_env_id)
    tasks = tasks[:max_tasks]

    model_rollouts = []
    baseline_rollouts = []
    for task in tasks:
        model_rollouts.append(
            await rollout_model_on_task(
                model=model,
                split=split,
                task_id=task.task_spec["task_id"],
                openreward_env_id=openreward_env_id,
                temperature=temperature,
                max_tool_calls=max_tool_calls,
                max_completion_tokens=max_completion_tokens,
            )
        )
        baseline_rollouts.append(
            await run_policy_rollout(
                baseline_name="weather_aware_rotation",
                split=split,
                task_id=task.task_spec["task_id"],
                openreward_env_id=openreward_env_id,
            )
        )

    summary = {
        **PIPELINE_META,
        "split": split,
        "openreward_env_id": openreward_env_id,
        "temperature": temperature,
        "max_tool_calls": max_tool_calls,
        "max_completion_tokens": max_completion_tokens,
        "inference_name": model.get_inference_name(),
        "model": summarize_model_rollouts(model_rollouts),
        "weather_aware_rotation": summarize_policy_rollouts(baseline_rollouts),
        "tasks": [task.task_spec["task_id"] for task in tasks],
    }
    await backend.close()
    return summary


@app.local_entrypoint()
def main(
    split: str = EVAL_DEFAULTS.split,
    max_tasks: int = 16,
    openreward_env_id: str = PIPELINE_META["openreward_env_id"],
    temperature: float = EVAL_DEFAULTS.temperature,
    max_tool_calls: int = EVAL_DEFAULTS.max_tool_calls,
    max_completion_tokens: int = EVAL_DEFAULTS.max_completion_tokens,
) -> None:
    summary = run_eval_compare.remote(
        split=split,
        max_tasks=max_tasks,
        openreward_env_id=openreward_env_id,
        temperature=temperature,
        max_tool_calls=max_tool_calls,
        max_completion_tokens=max_completion_tokens,
    )
    dump_json(comparison_path(split), summary)
    print(json.dumps(summary, indent=2))
