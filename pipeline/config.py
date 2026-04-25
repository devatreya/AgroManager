from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import (
    DEFAULT_MAX_TOOL_CALLS,
    MODEL_NAME,
    MODAL_VOLUME_MOUNTS,
    OPENREWARD_ENV_ID,
    PROJECT_NAME,
    QWEN_MODEL_NAME,
    TOTAL_QUARTERS,
    project_root,
)


ARTIFACTS_DIR = project_root() / "artifacts"
SFT_DIR = ARTIFACTS_DIR / "sft"
TRAINING_DIR = project_root() / "training"
EVAL_DIR = project_root() / "eval"


@dataclass(frozen=True)
class SFTDefaults:
    epochs: int = 1
    batch_size: int = 4
    peak_lr: float = 2e-4
    warmup_ratio: float = 0.1
    schedule: str = "cosine"


@dataclass(frozen=True)
class RLDefaults:
    split: str = "train"
    validation_split: str = "validation"
    train_steps: int = 8
    groups_per_step: int = 4
    trajectories_per_group: int = 4
    learning_rate: float = 1e-5
    loss_function: str = "cispo"
    kl_penalty: float = 0.0
    eval_every: int = 2
    temperature: float = 0.2
    seed: int = 23
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_completion_tokens: int = 256


@dataclass(frozen=True)
class EvalDefaults:
    split: str = "validation"
    temperature: float = 0.0
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_completion_tokens: int = 256


SFT_DEFAULTS = SFTDefaults()
RL_DEFAULTS = RLDefaults()
EVAL_DEFAULTS = EvalDefaults()


def art_internal_model_config(mode: str = "shared", seed: int = RL_DEFAULTS.seed) -> dict:
    config = {
        "init_args": {
            "max_seq_length": 32_768,
            "gpu_memory_utilization": 0.88,
            "random_state": seed,
            "fast_inference": mode == "shared",
        },
        "trainer_args": {
            "gradient_accumulation_steps": 1,
            "per_device_train_batch_size": 1,
        },
    }
    if mode == "dedicated":
        config["trainer_gpu_ids"] = [0]
        config["inference_gpu_ids"] = [1]
    return config


def training_summary_path(kind: str) -> Path:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINING_DIR / f"{kind}_result.json"


def comparison_path(split: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return EVAL_DIR / f"{split}_comparison.json"


def modal_results_path(kind: str) -> str:
    return f"{MODAL_VOLUME_MOUNTS['results']}/{kind}.json"


PIPELINE_META = {
    "project_name": PROJECT_NAME,
    "model_name": MODEL_NAME,
    "base_model": QWEN_MODEL_NAME,
    "openreward_env_id": OPENREWARD_ENV_ID,
    "total_quarters": TOTAL_QUARTERS,
}
