from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from config import MODAL_ENV, MODAL_VOLUME_MOUNTS, MODAL_VOLUME_NAMES, PROJECT_NAME, project_root


REPO_ROOT = project_root()


def build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "procps")
        .add_local_file(str(REPO_ROOT / "requirements.modal.txt"), "/root/project/requirements.modal.txt", copy=True)
        .run_commands("pip install -r /root/project/requirements.modal.txt")
        .add_local_dir(str(REPO_ROOT), "/root/project", copy=True)
        .env(MODAL_ENV)
    )


def build_volumes() -> dict[str, modal.Volume]:
    return {
        MODAL_VOLUME_MOUNTS["art"]: modal.Volume.from_name(
            MODAL_VOLUME_NAMES["art"], create_if_missing=True
        ),
        MODAL_VOLUME_MOUNTS["hf_cache"]: modal.Volume.from_name(
            MODAL_VOLUME_NAMES["hf_cache"], create_if_missing=True
        ),
        MODAL_VOLUME_MOUNTS["results"]: modal.Volume.from_name(
            MODAL_VOLUME_NAMES["results"], create_if_missing=True
        ),
    }


def build_secrets() -> list[modal.Secret]:
    return [
        modal.Secret.from_name("HF_TOKEN"),
        modal.Secret.from_name("WANDB_API_KEY"),
        modal.Secret.from_name("OPENREWARD_API_KEY"),
    ]


def build_app(name_suffix: str) -> tuple[modal.App, modal.Image, dict[str, modal.Volume], list[modal.Secret]]:
    image = build_image()
    volumes = build_volumes()
    secrets = build_secrets()
    app = modal.App(f"{PROJECT_NAME}-{name_suffix}", image=image)
    return app, image, volumes, secrets


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
