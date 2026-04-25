from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import PIPELINE_META, SFT_DEFAULTS, training_summary_path
from pipeline.modal_common import build_app, dump_json

app, image, volumes, secrets = build_app("train-sft")


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

    backend = LocalBackend(path=MODAL_VOLUME_MOUNTS["art"])
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=QWEN_MODEL_NAME,
        base_path=MODAL_VOLUME_MOUNTS["art"],
        _internal_config=art_internal_model_config(mode="shared"),
    )
    await model.register(backend)
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
    summary = {
        **PIPELINE_META,
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "step": step,
        "epochs": SFT_DEFAULTS.epochs,
        "batch_size": SFT_DEFAULTS.batch_size,
        "peak_lr": SFT_DEFAULTS.peak_lr,
        "schedule": SFT_DEFAULTS.schedule,
        "warmup_ratio": SFT_DEFAULTS.warmup_ratio,
        "inference_name": model.get_inference_name(),
    }
    await backend.close()
    return summary


@app.local_entrypoint()
def main() -> None:
    summary = run_train_sft.remote()
    dump_json(training_summary_path("sft"), summary)
    print(json.dumps(summary, indent=2))
