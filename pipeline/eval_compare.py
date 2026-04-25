from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.config import EVAL_DEFAULTS, PIPELINE_META, comparison_path
from pipeline.modal_common import build_app, dump_json

app, image, volumes, secrets = build_app("eval-compare")


def summarize(rollouts: list[dict]) -> dict[str, float]:
    count = max(1, len(rollouts))
    return {
        "mean_terminal_cash": sum(r["final_state"]["cash"] for r in rollouts) / count,
        "mean_final_soil_health": sum(r["mean_final_soil"] for r in rollouts) / count,
        "bankruptcy_rate": sum(1 for r in rollouts if r["ever_bankrupt"]) / count,
        "completion_rate": sum(r["completion_rate"] for r in rollouts) / count,
        "mean_total_episode_reward": sum(r["total_reward"] for r in rollouts) / count,
        "mean_terminal_score": sum(r["terminal_score"] for r in rollouts) / count,
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
                temperature=EVAL_DEFAULTS.temperature,
                max_tool_calls=EVAL_DEFAULTS.max_tool_calls,
                max_completion_tokens=EVAL_DEFAULTS.max_completion_tokens,
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
        "model": summarize(model_rollouts),
        "weather_aware_rotation": summarize(baseline_rollouts),
        "tasks": [task.task_spec["task_id"] for task in tasks],
    }
    await backend.close()
    return summary


@app.local_entrypoint()
def main(
    split: str = EVAL_DEFAULTS.split,
    max_tasks: int = 16,
    openreward_env_id: str = PIPELINE_META["openreward_env_id"],
) -> None:
    summary = run_eval_compare.remote(
        split=split,
        max_tasks=max_tasks,
        openreward_env_id=openreward_env_id,
    )
    dump_json(comparison_path(split), summary)
    print(json.dumps(summary, indent=2))
