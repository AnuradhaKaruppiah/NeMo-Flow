# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Mini SWE Agent's native trajectory format into NeMo Relay events.

This adapter is intentionally based only on the trajectory that Mini SWE Agent
already writes. It is useful for integrations where Mini SWE Agent cannot be
modified to call Relay-owned observer hooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import nemo_relay
from nemo_relay.integrations.mini_swe_agent.observer import (
    MiniSweAgentObservabilityConfig,
    NemoRelayMiniSweObserver,
)


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item.get("output") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _tool_call_key(tool_call: dict[str, Any]) -> str | None:
    return tool_call.get("id") or tool_call.get("call_id") or tool_call.get("tool_call_id")


def _tool_calls_by_id(message: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tool_calls: dict[str, dict[str, Any]] = {}
    for tool_call in message.get("tool_calls") or []:
        if isinstance(tool_call, dict) and (key := _tool_call_key(tool_call)):
            tool_calls[str(key)] = tool_call

    response = message.get("extra", {}).get("response")
    if isinstance(response, dict):
        choices = response.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_message = choice.get("message") or {}
            if not isinstance(choice_message, dict):
                continue
            for tool_call in choice_message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and (key := _tool_call_key(tool_call)):
                    tool_calls.setdefault(str(key), tool_call)

    for item in message.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        key = item.get("call_id") or item.get("id")
        if key:
            tool_calls.setdefault(str(key), item)
    return tool_calls


def _json_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed
    return value


def _enrich_action(action: Any, tool_calls: dict[str, dict[str, Any]]) -> Any:
    if not isinstance(action, dict):
        return action

    enriched = dict(action)
    tool_call_id = enriched.get("tool_call_id") or enriched.get("call_id") or enriched.get("id")
    tool_call = tool_calls.get(str(tool_call_id)) if tool_call_id else None
    if not isinstance(tool_call, dict):
        return enriched

    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = function.get("name") or tool_call.get("name")
    arguments = _json_arguments(function.get("arguments") or tool_call.get("arguments"))
    if name and "name" not in enriched:
        enriched["name"] = name
    if arguments and "arguments" not in enriched:
        enriched["arguments"] = arguments
    return enriched


def _actions(message: dict[str, Any]) -> list[Any]:
    actions = message.get("extra", {}).get("actions") or []
    if not isinstance(actions, list):
        return []
    tool_calls = _tool_calls_by_id(message)
    return [_enrich_action(action, tool_calls) for action in actions]


def _is_assistant_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "assistant" or message.get("object") == "response"


def _is_observation_message(message: dict[str, Any]) -> bool:
    return message.get("role") == "tool" or message.get("type") == "function_call_output"


def _observation_call_id(message: dict[str, Any]) -> str | None:
    for key in ("tool_call_id", "call_id", "id", "tool_use_id"):
        if value := message.get(key):
            return str(value)
    return None


def _observation_result(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
    output = extra.get("raw_output")
    if output is None:
        output = message.get("output")
    if output is None:
        output = _message_text(message.get("content"))

    result = {"output": output}
    if "returncode" in extra:
        result["returncode"] = extra.get("returncode")
    if "exception_info" in extra:
        result["exception_info"] = extra.get("exception_info")
    return result


def _paired_observations(actions: list[Any], observations: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    observations_by_id = {
        call_id: observation
        for observation in observations
        if (call_id := _observation_call_id(observation)) is not None
    }
    unpaired = iter(observations)
    paired: list[dict[str, Any] | None] = []
    for action in actions:
        action_id = None
        if isinstance(action, dict):
            for key in ("tool_call_id", "call_id", "id", "tool_use_id"):
                if value := action.get(key):
                    action_id = str(value)
                    break
        if action_id and action_id in observations_by_id:
            paired.append(observations_by_id[action_id])
        else:
            paired.append(next(unpaired, None))
    return paired


def replay_trajectory(
    trajectory: dict[str, Any],
    observer: NemoRelayMiniSweObserver,
    *,
    task: str | None = None,
) -> None:
    """Replay a Mini SWE Agent native trajectory into an initialized observer."""

    messages = trajectory.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    initial_messages: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {"system", "user"}:
            initial_messages.append(message)
            continue
        break

    if task is None and initial_messages:
        task = _message_text(initial_messages[-1].get("content"))

    observer.on_run_start(task=task or "", kwargs={"source": "mini_swe_native_trajectory"}, messages=initial_messages)

    step_index = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict) or not _is_assistant_message(message):
            index += 1
            continue

        step_index += 1
        request_messages = [item for item in messages[:index] if isinstance(item, dict)]
        observations: list[dict[str, Any]] = []
        next_index = index + 1
        while next_index < len(messages):
            candidate = messages[next_index]
            if isinstance(candidate, dict) and _is_observation_message(candidate):
                observations.append(candidate)
                next_index += 1
                continue
            break

        observer.on_step_start(step_index=step_index, messages=request_messages)
        observer.on_model_start(call_index=step_index, messages=request_messages)
        observer.on_model_end(
            call_index=step_index,
            message=message,
            cost=message.get("extra", {}).get("cost"),
            total_cost=trajectory.get("info", {}).get("model_stats", {}).get("instance_cost"),
            messages=[*request_messages, message],
        )

        actions = _actions(message)
        paired_observations = _paired_observations(actions, observations)
        for action_index, action in enumerate(actions):
            observer.on_action_start(action_index=action_index, action=action, message=message)
            observation = paired_observations[action_index]
            observer.on_action_end(
                action_index=action_index,
                action=action,
                output=_observation_result(observation) if observation else None,
                message=message,
            )

        observer.on_step_end(
            step_index=step_index,
            message=message,
            observations=observations,
            messages=[item for item in messages[:next_index] if isinstance(item, dict)],
        )
        index = next_index

    exit_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "exit"]
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    exit_status = info.get("exit_status") or (
        exit_messages[-1].get("extra", {}).get("exit_status") if exit_messages else ""
    )
    submission = info.get("submission") or (
        exit_messages[-1].get("extra", {}).get("submission") if exit_messages else ""
    )
    if exit_status == "Submitted" or submission:
        observer.on_submit(
            exception=RuntimeError("Submitted"),
            messages=exit_messages,
            submission=str(submission or ""),
        )
    observer.on_run_end(result=info, messages=[item for item in messages if isinstance(item, dict)])


def export_trajectory(
    trajectory: dict[str, Any],
    config: MiniSweAgentObservabilityConfig,
    *,
    task: str | None = None,
) -> dict[str, str]:
    """Export Mini SWE Agent's native trajectory as Relay ATOF and ATIF artifacts."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = config.trajectory_id or f"mini-swe-agent-{uuid4()}"
    atof_path = output_dir / config.atof_filename
    atif_filename = config.atif_filename_template.replace("{session_id}", session_id)
    atif_path = output_dir / atif_filename

    atof_config = nemo_relay.AtofExporterConfig()
    atof_config.output_directory = str(output_dir)
    atof_config.filename = config.atof_filename
    atof_config.mode = nemo_relay.AtofExporterMode.Overwrite
    atof_exporter = nemo_relay.AtofExporter(atof_config)
    trajectory_info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    agent_version = config.agent_version or trajectory_info.get("mini_version") or "unknown"
    atif_exporter = nemo_relay.AtifExporter(
        session_id,
        config.agent_name,
        str(agent_version),
        model_name=config.model_name,
        tool_definitions=config.tool_definitions,
        extra=config.metadata(),
    )
    subscriber_suffix = uuid4().hex
    atof_subscriber = f"mini_swe_agent_trajectory_atof_{subscriber_suffix}"
    atif_subscriber = f"mini_swe_agent_trajectory_atif_{subscriber_suffix}"
    atof_exporter.register(atof_subscriber)
    atif_exporter.register(atif_subscriber)

    observer = NemoRelayMiniSweObserver(
        instance_id=config.instance_id,
        model_name=config.model_name,
        task_index=config.task_index,
        rollout_index=config.rollout_index,
        extra=config.metadata(),
    )
    try:
        replay_trajectory(trajectory, observer, task=task)
    except Exception:
        observer.close("trajectory_replay_error")
        raise
    finally:
        nemo_relay.subscribers.flush()
        atof_exporter.force_flush()
        atif_path.write_text(atif_exporter.export_json(), encoding="utf-8")
        atif_exporter.deregister(atif_subscriber)
        atof_exporter.deregister(atof_subscriber)
        atof_exporter.shutdown()

    artifacts: dict[str, str] = {}
    if atof_path.exists():
        artifacts["atof"] = str(atof_path)
    if atif_path.exists():
        artifacts["atif"] = str(atif_path)
    return artifacts


__all__ = ["export_trajectory", "replay_trajectory"]
