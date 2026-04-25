from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.config import PIPELINE_META, RL_DEFAULTS, training_summary_path
from pipeline.modal_common import build_app, dump_json

app, image, volumes, secrets = build_app("train-rl")


def _numeric_metrics(result: dict) -> dict[str, float | int | bool]:
    metrics = {}
    for key, value in result.items():
        if isinstance(value, (float, int, bool)):
            metrics[key] = value
    return metrics


@app.function(
    gpu="H100:2",
    timeout=60 * 60 * 12,
    volumes=volumes,
    secrets=secrets,
)
async def run_train_rl(
    openreward_env_id: str,
    art_mode: str = "shared",
) -> dict:
    import art
    from art.local import LocalBackend

    from config import MODEL_NAME, MODAL_VOLUME_MOUNTS, PROJECT_NAME, QWEN_MODEL_NAME
    from pipeline.art_rollout import rollout_model_on_task
    from pipeline.config import art_internal_model_config
    from pipeline.farm_session import HostedFarmSession

    backend = LocalBackend(path=MODAL_VOLUME_MOUNTS["art"])
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=QWEN_MODEL_NAME,
        base_path=MODAL_VOLUME_MOUNTS["art"],
        _internal_config=art_internal_model_config(mode=art_mode, seed=RL_DEFAULTS.seed),
    )
    await model.register(backend)

    train_tasks = await HostedFarmSession.list_tasks(RL_DEFAULTS.split, openreward_env_id=openreward_env_id)
    validation_tasks = await HostedFarmSession.list_tasks(
        RL_DEFAULTS.validation_split,
        openreward_env_id=openreward_env_id,
    )

    step_summaries = []
    for step_index in range(RL_DEFAULTS.train_steps):
        task_slice_start = (step_index * RL_DEFAULTS.groups_per_step) % len(train_tasks)
        selected = train_tasks[task_slice_start : task_slice_start + RL_DEFAULTS.groups_per_step]
        if len(selected) < RL_DEFAULTS.groups_per_step:
            selected.extend(train_tasks[: RL_DEFAULTS.groups_per_step - len(selected)])

        groups = []
        for task in selected:
            trajectories = []
            for _ in range(RL_DEFAULTS.trajectories_per_group):
                rollout = await rollout_model_on_task(
                    model=model,
                    split=RL_DEFAULTS.split,
                    task_id=task.task_spec["task_id"],
                    openreward_env_id=openreward_env_id,
                    temperature=RL_DEFAULTS.temperature,
                    max_tool_calls=RL_DEFAULTS.max_tool_calls,
                    max_completion_tokens=RL_DEFAULTS.max_completion_tokens,
                )
                trajectory = rollout["trajectory"]
                trajectory.reward = float(rollout["terminal_score"])
                trajectory.metrics.update(_numeric_metrics(rollout))
                trajectories.append(trajectory.finish())
            groups.append(
                art.TrajectoryGroup(
                    trajectories,
                    metadata={"task_id": task.task_spec["task_id"]},
                )
            )

        await model.log(groups, split="train")
        train_result = await backend.train(
            model,
            groups,
            learning_rate=RL_DEFAULTS.learning_rate,
            kl_penalty_coef=RL_DEFAULTS.kl_penalty,
            allow_training_without_logprobs=True,
        )
        await model.log(metrics=train_result.metrics, step=train_result.step, split="train")

        step_summary = {
            "train_step": step_index + 1,
            "checkpoint_step": train_result.step,
            "metrics": train_result.metrics,
        }

        if (step_index + 1) % RL_DEFAULTS.eval_every == 0:
            eval_rollouts = []
            for task in validation_tasks[:4]:
                eval_rollouts.append(
                    await rollout_model_on_task(
                        model=model,
                        split=RL_DEFAULTS.validation_split,
                        task_id=task.task_spec["task_id"],
                        openreward_env_id=openreward_env_id,
                        temperature=0.0,
                        max_tool_calls=RL_DEFAULTS.max_tool_calls,
                        max_completion_tokens=RL_DEFAULTS.max_completion_tokens,
                    )
                )
            step_summary["validation_mean_terminal_score"] = sum(
                rollout["terminal_score"] for rollout in eval_rollouts
            ) / max(1, len(eval_rollouts))
            await model.log(
                metrics={"terminal_score": step_summary["validation_mean_terminal_score"]},
                split="val",
                step=train_result.step,
            )
        step_summaries.append(step_summary)

    final_step = await model.get_step()
    summary = {
        **PIPELINE_META,
        "openreward_env_id": openreward_env_id,
        "art_mode": art_mode,
        "step_summaries": step_summaries,
        "final_step": final_step,
        "inference_name": model.get_inference_name(),
        "loss_function": RL_DEFAULTS.loss_function,
        "allow_training_without_logprobs": True,
    }
    await backend.close()
    return summary


@app.local_entrypoint()
def main(
    openreward_env_id: str = PIPELINE_META["openreward_env_id"],
    art_mode: str = "shared",
) -> None:
    summary = run_train_rl.remote(openreward_env_id=openreward_env_id, art_mode=art_mode)
    dump_json(training_summary_path("rl"), summary)
    print(json.dumps(summary, indent=2))
