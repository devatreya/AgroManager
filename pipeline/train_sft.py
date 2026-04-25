from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import PIPELINE_META, SFT_DEFAULTS, training_summary_path
from pipeline.modal_common import build_app, dump_json

app, image, volumes, secrets = build_app("train-sft")


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _loss_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in rows:
        for key in ("loss/train", "train/loss/train"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    if not values:
        return {"available": False}
    return {
        "available": True,
        "records": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _model_output_dir(base_path: str, project: str, model_name: str) -> Path:
    return Path(base_path) / project / "models" / model_name


@app.function(
    gpu="H100:2",
    timeout=60 * 60 * 8,
    volumes=volumes,
    secrets=secrets,
)
async def run_train_sft() -> dict:
    import art
    from art.local import LocalBackend
    from art.utils.sft import train_sft_from_file

    from config import MODEL_NAME, MODAL_VOLUME_MOUNTS, PROJECT_NAME, QWEN_MODEL_NAME
    from pipeline.config import art_internal_model_config

    train_file = Path("/root/project/artifacts/sft/train.jsonl")
    validation_file = Path("/root/project/artifacts/sft/validation.jsonl")

    if not train_file.exists():
        raise FileNotFoundError(
            f"{train_file} was not found. Run scripts/prepare_sft_data.py first."
        )
    if not validation_file.exists():
        raise FileNotFoundError(
            f"{validation_file} was not found. Run scripts/prepare_sft_data.py first."
        )

    backend = LocalBackend(path=MODAL_VOLUME_MOUNTS["art"])
    sft_internal_config = art_internal_model_config(mode="shared")
    sft_internal_config["init_args"]["fast_inference"] = False
    sft_internal_config["init_args"]["max_seq_length"] = 8192
    sft_internal_config.setdefault("engine_args", {})
    sft_internal_config["engine_args"].update(
        {
            "gpu_memory_utilization": 0.5,
            "max_model_len": 8192,
        }
    )
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=QWEN_MODEL_NAME,
        base_path=MODAL_VOLUME_MOUNTS["art"],
        _internal_config=sft_internal_config,
    )
    await model.register(backend)

    train_examples = _count_jsonl_rows(train_file)
    validation_examples = _count_jsonl_rows(validation_file)

    await train_sft_from_file(
        model=model,
        file_path=str(train_file),
        epochs=SFT_DEFAULTS.epochs,
        batch_size=SFT_DEFAULTS.batch_size,
        peak_lr=SFT_DEFAULTS.peak_lr,
        schedule_type=SFT_DEFAULTS.schedule,
        warmup_ratio=SFT_DEFAULTS.warmup_ratio,
        verbose=True,
    )
    step = await model.get_step()

    output_dir = _model_output_dir(MODAL_VOLUME_MOUNTS["art"], PROJECT_NAME, MODEL_NAME)
    checkpoint_path = output_dir / "checkpoints" / f"{step:04d}"
    train_loss_summary = _loss_summary(_history_rows(output_dir / "history.jsonl"))

    summary = {
        **PIPELINE_META,
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "step": step,
        "epochs": SFT_DEFAULTS.epochs,
        "batch_size": SFT_DEFAULTS.batch_size,
        "peak_lr": SFT_DEFAULTS.peak_lr,
        "schedule": SFT_DEFAULTS.schedule,
        "warmup_ratio": SFT_DEFAULTS.warmup_ratio,
        "inference_name": model.get_inference_name(),
        "checkpoint_identifier": model.get_inference_name(step=step),
        "checkpoint_path": str(checkpoint_path),
        "train_loss_summary": train_loss_summary,
        "validation_loss_summary": {
            "available": False,
            "method": "skipped_for_hackathon_runtime",
        },
    }
    await backend.close()
    return summary


@app.local_entrypoint()
def main() -> None:
    summary = run_train_sft.remote()
    dump_json(training_summary_path("sft"), summary)
    print(json.dumps(summary, indent=2))
