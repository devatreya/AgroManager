from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from openreward import AsyncOpenReward

from config import COMMIT_TOOL_NAME, OPENREWARD_ENV_ID


TOOL_RESULT_STATE_RE = re.compile(r"Cash:\s*GBP\s*([0-9\-.]+)", re.IGNORECASE)
TOOL_RESULT_TERMINAL_RE = re.compile(r"Terminal score:\s*([0-9.]+)", re.IGNORECASE)
TOOL_RESULT_QUARTER_RE = re.compile(r"Quarter\s+(\d+)", re.IGNORECASE)


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.replace("£", "GBP").split())


def _replace_currency_symbols(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("£", "GBP")
    if isinstance(value, list):
        return [_replace_currency_symbols(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_currency_symbols(item) for key, item in value.items()}
    return value


def _strip_schema_noise(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_schema_noise(item) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"title", "$schema", "$defs"}:
                continue
            cleaned[key] = _strip_schema_noise(item)
        return cleaned
    return value


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = copy.deepcopy(schema.get("$defs", {}))

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported schema ref {ref!r}")
            target_name = ref.rsplit("/", 1)[-1]
            target = copy.deepcopy(defs[target_name])
            merged = {key: value for key, value in node.items() if key != "$ref"}
            target.update(merged)
            return resolve(target)
        return {key: resolve(value) for key, value in node.items() if key != "$defs"}

    return resolve(copy.deepcopy(schema))


def _compact_commit_plan_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    compacted = {
        "type": "object",
        "description": _collapse_whitespace(schema.get("description", "")),
        "properties": {
            "capital_action": properties.get("capital_action", {}),
        },
        "required": list(schema.get("required", [])),
        "additionalProperties": False,
    }
    for plot_name in ("plot_1", "plot_2", "plot_3", "plot_4"):
        plot_schema = copy.deepcopy(properties.get(plot_name, {}))
        if plot_schema:
            plot_schema.setdefault("type", "object")
            plot_schema["description"] = _collapse_whitespace(
                plot_schema.get("description", f"{plot_name} plan")
            )
            compacted["properties"][plot_name] = plot_schema
    return compacted


def normalize_tool_schema(schema: dict[str, Any] | None, tool_name: str) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}, "additionalProperties": False}
    normalized = _replace_currency_symbols(_inline_local_refs(copy.deepcopy(schema)))
    normalized = _strip_schema_noise(normalized)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    normalized.setdefault("additionalProperties", False)
    normalized["description"] = _collapse_whitespace(normalized.get("description", ""))
    if tool_name == COMMIT_TOOL_NAME:
        normalized = _compact_commit_plan_schema(normalized)
    return normalized


def normalize_tool_specs(tool_specs: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for spec in tool_specs:
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": _collapse_whitespace(spec.description or ""),
                    "parameters": normalize_tool_schema(spec.input_schema, spec.name),
                },
            }
        )
    return normalized


def parse_tool_text_fallback(tool_name: str, text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"tool": tool_name}
    quarter_match = TOOL_RESULT_QUARTER_RE.search(text)
    if quarter_match:
        parsed.setdefault("state", {})["quarter_hint"] = int(quarter_match.group(1))
    cash_match = TOOL_RESULT_STATE_RE.search(text)
    if cash_match:
        parsed.setdefault("state", {})["cash"] = float(cash_match.group(1))
    terminal_match = TOOL_RESULT_TERMINAL_RE.search(text)
    if terminal_match:
        parsed["episode_metrics"] = {"terminal_score": float(terminal_match.group(1))}
    return parsed


@dataclass
class HostedToolResult:
    tool_name: str
    text: str
    metadata: dict[str, Any]
    reward: float | None
    finished: bool
    state: dict[str, Any] | None
    episode_metrics: dict[str, Any] | None


class HostedFarmSession:
    _client: AsyncOpenReward | None = None
    _environments: dict[str, Any] = {}
    _tasks: dict[tuple[str, str], list[Any]] = {}
    _tool_specs: dict[str, list[dict[str, Any]]] = {}

    def __init__(self, split: str, task_id: str, openreward_env_id: str = OPENREWARD_ENV_ID) -> None:
        self.split = split
        self.task_id = task_id
        self.openreward_env_id = openreward_env_id
        self.task = None
        self.environment = None
        self.session = None
        self.prompt = ""
        self.last_state: dict[str, Any] | None = None
        self.last_episode_metrics: dict[str, Any] | None = None

    @classmethod
    def reset_caches(cls) -> None:
        cls._environments.clear()
        cls._tasks.clear()
        cls._tool_specs.clear()

    @classmethod
    def _get_client(cls) -> AsyncOpenReward:
        if cls._client is None:
            cls._client = AsyncOpenReward()
        return cls._client

    @classmethod
    async def environment_for(cls, env_id: str) -> Any:
        if env_id not in cls._environments:
            cls._environments[env_id] = cls._get_client().environments.get(env_id)
        return cls._environments[env_id]

    @classmethod
    async def list_tasks(cls, split: str, openreward_env_id: str = OPENREWARD_ENV_ID) -> list[Any]:
        cache_key = (openreward_env_id, split)
        if cache_key not in cls._tasks:
            env = await cls.environment_for(openreward_env_id)
            cls._tasks[cache_key] = list(await env.list_tasks(split))
        return cls._tasks[cache_key]

    @classmethod
    async def resolve_task(
        cls,
        split: str,
        task_id: str,
        openreward_env_id: str = OPENREWARD_ENV_ID,
    ) -> Any:
        for task in await cls.list_tasks(split, openreward_env_id=openreward_env_id):
            if task.task_spec["task_id"] == task_id:
                return task
        raise KeyError(f"Task {task_id!r} not found in split {split!r}")

    @classmethod
    async def hosted_tool_specs(cls, openreward_env_id: str = OPENREWARD_ENV_ID) -> list[dict[str, Any]]:
        if openreward_env_id not in cls._tool_specs:
            env = await cls.environment_for(openreward_env_id)
            raw_specs = await env.list_tools()
            cls._tool_specs[openreward_env_id] = normalize_tool_specs(raw_specs)
        return copy.deepcopy(cls._tool_specs[openreward_env_id])

    async def __aenter__(self) -> "HostedFarmSession":
        self.environment = await self.environment_for(self.openreward_env_id)
        self.task = await self.resolve_task(
            self.split,
            self.task_id,
            openreward_env_id=self.openreward_env_id,
        )
        self.session = self.environment.session(task=self.task)
        await self.session.__aenter__()
        prompt_blocks = await self.session.get_prompt()
        self.prompt = "\n\n".join(
            block.text for block in prompt_blocks if getattr(block, "type", "text") == "text"
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.__aexit__(exc_type, exc, tb)

    async def get_prompt(self) -> str:
        return self.prompt

    async def get_tool_specs(self) -> list[dict[str, Any]]:
        return await self.hosted_tool_specs(self.openreward_env_id)

    async def call_tool(self, tool_name: str, payload: dict[str, Any] | None = None) -> HostedToolResult:
        if self.session is None:
            raise RuntimeError("HostedFarmSession must be entered before use")
        payload = payload or {}
        output = await self.session.call_tool(tool_name, payload)
        text = "\n".join(
            block.text for block in output.blocks if getattr(block, "type", "text") == "text"
        )
        metadata = dict(output.metadata or {})
        parsed = parse_tool_text_fallback(tool_name, text)

        state = metadata.get("state")
        if state is None:
            state = parsed.get("state")
        episode_metrics = metadata.get("episode_metrics")
        if episode_metrics is None:
            episode_metrics = parsed.get("episode_metrics")

        if state is not None:
            self.last_state = dict(state)
        if episode_metrics is not None:
            self.last_episode_metrics = dict(episode_metrics)

        return HostedToolResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            reward=output.reward,
            finished=output.finished,
            state=self.last_state,
            episode_metrics=self.last_episode_metrics,
        )
