# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Mini SWE Agent observer integration."""

from __future__ import annotations

import asyncio
import json

import nemo_relay
import nemo_relay.integrations.mini_swe_agent.observer as observer_module
from nemo_relay import MarkEvent, ScopeEvent, subscribers
from nemo_relay.integrations.mini_swe_agent import (
    MiniSweAgentObservabilityConfig,
    NemoRelayMiniSweObserver,
    build_observability_config,
    replay_trajectory,
)


def _event_names(events: list[nemo_relay.Event]) -> list[str]:
    return [event.name for event in events]


def test_observer_emits_run_step_model_tool_and_submit_events(subscribed_events: list[nemo_relay.Event]) -> None:
    observer = NemoRelayMiniSweObserver(instance_id="django__django-13741", model_name="nvidia/test-model")

    initial_messages = [{"role": "user", "content": "fix the bug"}]
    assistant_message = {
        "role": "assistant",
        "content": "Inspect",
        "extra": {"actions": [{"command": "sed -n '1,40p' file.py"}]},
    }
    action = {"command": "sed -n '1,40p' file.py"}
    output = {"returncode": 0, "output": "class Example: pass"}

    observer.on_run_start(task="fix the bug", kwargs={}, messages=initial_messages)
    observer.on_step_start(step_index=1, messages=initial_messages)
    observer.on_model_start(call_index=1, messages=initial_messages)
    observer.on_model_end(
        call_index=1,
        message=assistant_message,
        cost=0.01,
        total_cost=0.01,
        messages=[*initial_messages, assistant_message],
    )
    observer.on_action_start(action_index=0, action=action, message=assistant_message)
    observer.on_action_end(action_index=0, action=action, output=output, message=assistant_message)
    observer.on_step_end(
        step_index=1,
        message=assistant_message,
        observations=[{"role": "user", "content": output["output"]}],
        messages=[*initial_messages, assistant_message],
    )
    observer.on_submit(
        exception=RuntimeError("submitted"),
        messages=[{"role": "exit", "extra": {"exit_status": "Submitted"}}],
        submission="patch",
    )
    observer.on_run_end(result={"exit_status": "Submitted"}, messages=[])
    subscribers.flush()

    names = _event_names(subscribed_events)
    assert "mini-swe-agent.run" in names
    assert "mini-swe-agent.step.1" in names
    assert "mini-swe-agent.model" in names
    assert "action.exec" in names
    assert "mini_swe_agent.submit" in names

    run_start = next(
        event
        for event in subscribed_events
        if isinstance(event, ScopeEvent) and event.name == "mini-swe-agent.run" and event.scope_category == "start"
    )
    assert run_start.category == "agent"
    assert run_start.metadata["instance_id"] == "django__django-13741"

    tool_end = next(
        event
        for event in subscribed_events
        if isinstance(event, ScopeEvent) and event.name == "action.exec" and event.scope_category == "end"
    )
    assert tool_end.category == "tool"
    assert tool_end.data == output

    submit_mark = next(event for event in subscribed_events if isinstance(event, MarkEvent))
    assert submit_mark.name == "mini_swe_agent.submit"
    assert submit_mark.data["submission"] == "patch"


def test_observer_uses_normalized_action_name(subscribed_events: list[nemo_relay.Event]) -> None:
    observer = NemoRelayMiniSweObserver(instance_id="django__django-13741", model_name="nvidia/test-model")

    observer.on_run_start(task="fix", kwargs={}, messages=[])
    observer.on_step_start(step_index=1, messages=[])
    observer.on_action_start(
        action_index=0,
        action={"name": "bash", "arguments": {"command": "echo hi"}, "id": "call-1", "command": "echo hi"},
        message={},
    )
    observer.on_action_end(
        action_index=0,
        action={"name": "bash", "arguments": {"command": "echo hi"}, "id": "call-1", "command": "echo hi"},
        output={"returncode": 0, "output": "hi"},
        message={},
    )
    observer.on_step_end(step_index=1, message={}, observations=[], messages=[])
    observer.on_run_end(result={"exit_status": "ok"}, messages=[])
    subscribers.flush()

    names = _event_names(subscribed_events)
    assert "bash" in names


def test_replay_trajectory_emits_events_from_native_mini_swe_messages(
    subscribed_events: list[nemo_relay.Event],
) -> None:
    trajectory = {
        "info": {
            "model_stats": {"instance_cost": 0.0, "api_calls": 1},
            "exit_status": "Submitted",
            "submission": "diff --git a/file.py b/file.py",
        },
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "I will inspect the file.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "sed -n '1,40p' file.py"}),
                        },
                    }
                ],
                "extra": {
                    "actions": [
                        {
                            "command": "sed -n '1,40p' file.py",
                            "tool_call_id": "call-1",
                        }
                    ]
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "<returncode>0</returncode>\n<output>class Example: pass</output>",
                "extra": {
                    "raw_output": "class Example: pass",
                    "returncode": 0,
                    "exception_info": "",
                },
            },
            {
                "role": "exit",
                "content": "diff --git a/file.py b/file.py",
                "extra": {"exit_status": "Submitted", "submission": "diff --git a/file.py b/file.py"},
            },
        ],
        "trajectory_format": "mini-swe-agent-1.1",
    }

    observer = NemoRelayMiniSweObserver(instance_id="django__django-13741", model_name="nvidia/test-model")
    replay_trajectory(trajectory, observer)
    subscribers.flush()

    names = _event_names(subscribed_events)
    assert "mini-swe-agent.run" in names
    assert "mini-swe-agent.step.1" in names
    assert "mini-swe-agent.model" in names
    assert "bash" in names
    assert "mini_swe_agent.submit" in names

    tool_start = next(
        event
        for event in subscribed_events
        if isinstance(event, ScopeEvent) and event.name == "bash" and event.scope_category == "start"
    )
    assert tool_start.data["command"] == "sed -n '1,40p' file.py"
    assert tool_start.data["name"] == "bash"
    assert tool_start.data["arguments"] == {"command": "sed -n '1,40p' file.py"}

    tool_end = next(
        event
        for event in subscribed_events
        if isinstance(event, ScopeEvent) and event.name == "bash" and event.scope_category == "end"
    )
    assert tool_end.data == {
        "output": "class Example: pass",
        "returncode": 0,
        "exception_info": "",
    }


def test_observer_close_finishes_open_handles(subscribed_events: list[nemo_relay.Event]) -> None:
    observer = NemoRelayMiniSweObserver(instance_id="django__django-13741", model_name="nvidia/test-model")
    observer.on_run_start(task="fix", kwargs={}, messages=[])
    observer.on_step_start(step_index=1, messages=[])
    observer.on_model_start(call_index=1, messages=[])
    observer.on_action_start(action_index=0, action={"command": "sleep 1"}, message={})

    observer.close("test-close")
    subscribers.flush()

    closed_events = [
        event
        for event in subscribed_events
        if isinstance(event, ScopeEvent) and event.scope_category == "end" and event.data
    ]
    payload = json.dumps([event.data for event in closed_events])
    assert "test-close" in payload


def test_build_observability_config_preserves_metadata() -> None:
    config = MiniSweAgentObservabilityConfig(
        output_dir="/tmp/relay",
        model_name="nvidia/model",
        trajectory_id="django-13741-run",
        instance_id="django__django-13741",
        task_index=2,
        rollout_index=4,
        extra={"gym_agent": "mini_swe_agent_2"},
    )

    observability = build_observability_config(config).to_dict()
    assert observability["atof"]["output_directory"] == "/tmp/relay"
    assert observability["atif"]["agent_name"] == "mini-swe-agent"
    assert observability["atif"]["model_name"] == "nvidia/model"
    assert observability["atif"]["extra"]["instance_id"] == "django__django-13741"
    assert observability["atif"]["extra"]["trajectory_id"] == "django-13741-run"
    assert observability["atif"]["extra"]["gym_agent"] == "mini_swe_agent_2"


def test_start_observability_can_run_inside_existing_event_loop(monkeypatch) -> None:
    sentinel = object()

    async def fake_start(config: MiniSweAgentObservabilityConfig) -> object:
        return sentinel

    monkeypatch.setattr(observer_module, "astart_observability", fake_start)

    async def call_start() -> object:
        return observer_module.start_observability(
            MiniSweAgentObservabilityConfig(output_dir="/tmp/relay", model_name="nvidia/model")
        )

    assert asyncio.run(call_start()) is sentinel
