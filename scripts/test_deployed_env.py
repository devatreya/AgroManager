from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import certifi
from packaging.version import Version

sys.path.append(str(Path(__file__).resolve().parents[1]))


def ensure_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required")
    if Version(package_version("openreward")) < Version("0.1.105"):
        raise SystemExit("openreward>=0.1.105 is required")
    cert_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cert_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_bundle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openreward-env-id", required=True)
    args = parser.parse_args()

    ensure_runtime()

    from openreward import OpenReward

    client = OpenReward()
    env = client.environments.get(args.openreward_env_id)
    tasks = env.list_tasks("validation")
    if not tasks:
        raise SystemExit("No validation tasks available on hosted environment")
    task = tasks[0]

    with env.session(task=task) as session:
        prompt = session.get_prompt()
        if not prompt:
            raise SystemExit("Hosted prompt was empty")
        state = session.call_tool("read_farm_state", {})
        if not state.metadata:
            raise SystemExit("read_farm_state returned no metadata")
        commit = session.call_tool(
            "commit_plan",
            {
                "capital_action": "none",
                "plot_1": {"crop": "field_beans", "fertiliser": "medium", "pest_control": "ipm"},
                "plot_2": {"crop": "barley", "fertiliser": "medium", "pest_control": "ipm"},
                "plot_3": {"crop": "wheat", "fertiliser": "medium", "pest_control": "ipm"},
                "plot_4": {"crop": "cover_crop", "fertiliser": "low", "pest_control": "none"},
            },
        )
        print(f"Hosted environment healthy. Reward={commit.reward} finished={commit.finished}")


if __name__ == "__main__":
    main()
