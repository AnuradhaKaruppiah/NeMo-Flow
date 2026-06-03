# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observer adapter that maps Mini SWE Agent lifecycle hooks to NeMo Relay."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import nemo_relay
from nemo_relay import plugin as relay_plugin
from nemo_relay.observability import AtifConfig, AtofConfig, ComponentSpec, ObservabilityConfig

_logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


@dataclass(slots=True)
class MiniSweAgentObservabilityConfig:
    """Configuration for Relay-backed Mini SWE Agent observer telemetry."""

    output_dir: str
    model_name: str = "unknown"
    trajectory_id: str | None = None
    instance_id: str | None = None
    task_index: Any = None
    rollout_index: Any = None
    agent_name: str = "mini-swe-agent"
    agent_version: str | None = None
    atof_filename: str = "events.atof.jsonl"
    atif_filename_template: str = "trajectory-{session_id}.atif.json"
    extra: dict[str, Any] = field(default_factory=dict)
    tool_definitions: list[dict[str, Any]] = field(default_factory=lambda: [{"name": "bash"}, {"name": "action.exec"}])

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "harness": "mini-swe-agent",
            "instance_id": self.instance_id,
            "trajectory_id": self.trajectory_id,
            "task_index": self.task_index,
            "rollout_index": self.rollout_index,
            "binding": "mini-swe-agent-observer",
        }
        metadata.update(self.extra)
        return _json_safe(metadata)


def build_observability_config(config: MiniSweAgentObservabilityConfig) -> ObservabilityConfig:
    """Build a Relay observability plugin config for Mini SWE Agent artifacts."""
    return ObservabilityConfig(
        atof=AtofConfig(
            enabled=True,
            output_directory=config.output_dir,
            filename=config.atof_filename,
            mode="overwrite",
        ),
        atif=AtifConfig(
            enabled=True,
            agent_name=config.agent_name,
            agent_version=config.agent_version,
            model_name=config.model_name,
            tool_definitions=_json_safe(config.tool_definitions),
            output_directory=config.output_dir,
            filename_template=config.atif_filename_template,
            extra=config.metadata(),
        ),
    )


def artifacts(output_dir: str) -> dict[str, str]:
    """Return generated Mini SWE Relay artifact paths, when present."""
    root = Path(output_dir)
    result: dict[str, str] = {}
    atof_path = root / "events.atof.jsonl"
    if atof_path.exists():
        result["atof"] = str(atof_path)
    atif_paths = sorted(root.glob("trajectory-*.atif.json"))
    if atif_paths:
        result["atif"] = str(atif_paths[-1])
    return result


class NemoRelayMiniSweObserver:
    """Mini SWE Agent observer that emits Relay scope, LLM, tool, and mark events."""

    def __init__(
        self,
        *,
        instance_id: str | None = None,
        model_name: str = "unknown",
        task_index: Any = None,
        rollout_index: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._model_name = model_name
        self._task_index = task_index
        self._rollout_index = rollout_index
        self._extra = dict(extra or {})
        self._agent_handle: Any = None
        self._active_step_index: int | None = None
        self._step_handles: dict[int, Any] = {}
        self._llm_handles: dict[int, Any] = {}
        self._tool_handles: dict[tuple[int | None, int], Any] = {}
        self._closed = False

    def _metadata(self, **extra: Any) -> dict[str, Any]:
        metadata = {
            "framework": "mini-swe-agent",
            "harness": "mini-swe-agent",
            "instance_id": self._instance_id,
            "task_index": self._task_index,
            "rollout_index": self._rollout_index,
        }
        metadata.update(self._extra)
        metadata.update(extra)
        return _json_safe(metadata)

    def _parent_handle(self) -> Any:
        if self._active_step_index is not None:
            return self._step_handles.get(self._active_step_index) or self._agent_handle
        return self._agent_handle

    def _event(self, name: str, data: Any = None, **metadata: Any) -> None:
        if self._agent_handle is None:
            return
        try:
            nemo_relay.scope.event(
                name,
                handle=self._parent_handle(),
                data=_json_safe(data or {}),
                metadata=self._metadata(**metadata),
            )
        except Exception:
            _logger.warning("NeMo Relay Mini SWE mark emit failed: %s", name, exc_info=True)

    @staticmethod
    def _tool_name(action: Any) -> str:
        if isinstance(action, dict):
            return str(
                action.get("name")
                or action.get("tool_name")
                or action.get("function_name")
                or action.get("command_name")
                or "action.exec"
            )
        return "action.exec"

    @staticmethod
    def _tool_call_id(action: Any) -> str | None:
        if not isinstance(action, dict):
            return None
        for key in ("tool_call_id", "call_id", "id", "tool_use_id"):
            value = action.get(key)
            if value:
                return str(value)
        return None

    def on_run_start(self, *, task: str, kwargs: dict[str, Any], messages: list[dict[str, Any]], **_: Any) -> None:
        if self._agent_handle is not None:
            return
        self._agent_handle = nemo_relay.scope.push(
            "mini-swe-agent.run",
            nemo_relay.ScopeType.Agent,
            input=_json_safe({"task": task, "messages": messages}),
            metadata=self._metadata(event="run_start", kwargs=kwargs),
        )

    def on_step_start(self, *, step_index: int, messages: list[dict[str, Any]], **_: Any) -> None:
        self._active_step_index = step_index
        self._step_handles[step_index] = nemo_relay.scope.push(
            f"mini-swe-agent.step.{step_index}",
            nemo_relay.ScopeType.Function,
            handle=self._agent_handle,
            input=_json_safe({"messages": messages}),
            metadata=self._metadata(event="step_start", step_index=step_index),
        )

    def on_model_start(self, *, call_index: int, messages: list[dict[str, Any]], **_: Any) -> None:
        request = nemo_relay.LLMRequest(
            {},
            _json_safe({"model": self._model_name, "messages": messages}),
        )
        self._llm_handles[call_index] = nemo_relay.llm.call(
            "mini-swe-agent.model",
            request,
            handle=self._parent_handle(),
            model_name=self._model_name,
            metadata=self._metadata(event="model_start", call_index=call_index),
        )

    def on_model_end(
        self,
        *,
        call_index: int,
        message: dict[str, Any] | None = None,
        cost: float | None = None,
        total_cost: float | None = None,
        messages: list[dict[str, Any]] | None = None,
        exception: BaseException | None = None,
        **_: Any,
    ) -> None:
        handle = self._llm_handles.pop(call_index, None)
        if handle is None:
            return
        response: dict[str, Any] = {
            "message": message,
            "cost": cost,
            "total_cost": total_cost,
            "messages": messages,
        }
        if exception is not None:
            response["exception"] = repr(exception)
        nemo_relay.llm.call_end(
            handle,
            _json_safe(response),
            metadata=self._metadata(
                event="model_end",
                call_index=call_index,
                status="error" if exception is not None else "ok",
            ),
        )

    def on_action_start(self, *, action_index: int, action: Any, message: dict[str, Any], **_: Any) -> None:
        key = (self._active_step_index, action_index)
        self._tool_handles[key] = nemo_relay.tools.call(
            self._tool_name(action),
            _json_safe(action),
            handle=self._parent_handle(),
            tool_call_id=self._tool_call_id(action),
            metadata=self._metadata(
                event="action_start", step_index=self._active_step_index, action_index=action_index
            ),
            data=_json_safe({"message": message}),
        )

    def on_action_end(
        self,
        *,
        action_index: int,
        action: Any,
        output: dict[str, Any] | None = None,
        message: dict[str, Any] | None = None,
        exception: BaseException | None = None,
        **_: Any,
    ) -> None:
        key = (self._active_step_index, action_index)
        handle = self._tool_handles.pop(key, None)
        if handle is None:
            return
        result: dict[str, Any] = output or {}
        if exception is not None:
            result = {"exception": repr(exception)}
        nemo_relay.tools.call_end(
            handle,
            _json_safe(result),
            metadata=self._metadata(
                event="action_end",
                step_index=self._active_step_index,
                action_index=action_index,
                status="error" if exception is not None else "ok",
            ),
            data=_json_safe({"action": action, "message": message}),
        )

    def on_step_end(
        self,
        *,
        step_index: int,
        message: dict[str, Any] | None = None,
        observations: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        exception: BaseException | None = None,
        **_: Any,
    ) -> None:
        handle = self._step_handles.pop(step_index, None)
        if handle is not None:
            output: dict[str, Any] = {
                "message": message,
                "observations": observations,
                "messages": messages,
            }
            if exception is not None:
                output["exception"] = repr(exception)
            nemo_relay.scope.pop(handle, output=_json_safe(output))
        if self._active_step_index == step_index:
            self._active_step_index = None

    def on_submit(
        self,
        *,
        exception: BaseException,
        messages: list[dict[str, Any]],
        submission: str,
        **_: Any,
    ) -> None:
        self._event(
            "mini_swe_agent.submit",
            {"submission": submission, "messages": messages, "exception": repr(exception)},
        )

    def on_interrupt(self, *, exception: BaseException, messages: list[dict[str, Any]], **_: Any) -> None:
        self._event("mini_swe_agent.interrupt", {"exception": repr(exception), "messages": messages})

    def on_error(self, *, exception: BaseException, messages: list[dict[str, Any]], **_: Any) -> None:
        self._event("mini_swe_agent.error", {"exception": repr(exception), "messages": messages})

    def on_run_end(self, *, result: dict[str, Any], messages: list[dict[str, Any]], **_: Any) -> None:
        self.close("run_end", result={"result": result, "messages": messages})

    def close(self, reason: str = "close", *, result: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True

        for call_index, handle in list(self._llm_handles.items()):
            try:
                nemo_relay.llm.call_end(
                    handle,
                    {"closed": True, "reason": reason},
                    metadata=self._metadata(event="model_close", call_index=call_index, status="closed"),
                )
            except Exception:
                _logger.debug("NeMo Relay Mini SWE LLM close failed", exc_info=True)
        self._llm_handles.clear()

        for (step_index, action_index), handle in list(self._tool_handles.items()):
            try:
                nemo_relay.tools.call_end(
                    handle,
                    {"closed": True, "reason": reason},
                    metadata=self._metadata(
                        event="action_close",
                        step_index=step_index,
                        action_index=action_index,
                        status="closed",
                    ),
                )
            except Exception:
                _logger.debug("NeMo Relay Mini SWE tool close failed", exc_info=True)
        self._tool_handles.clear()

        for step_index, handle in list(self._step_handles.items()):
            try:
                nemo_relay.scope.pop(handle, output={"closed": True, "reason": reason, "step_index": step_index})
            except Exception:
                _logger.debug("NeMo Relay Mini SWE step close failed", exc_info=True)
        self._step_handles.clear()
        self._active_step_index = None

        if self._agent_handle is not None:
            try:
                nemo_relay.scope.pop(
                    self._agent_handle,
                    output=_json_safe(result or {"closed": True, "reason": reason}),
                )
            finally:
                self._agent_handle = None


@dataclass(slots=True)
class MiniSweAgentRelayRun:
    """Closeable Mini SWE observer run returned by ``start_observability``."""

    observer: NemoRelayMiniSweObserver
    output_dir: str
    _clear: Callable[[], None]
    _closed: bool = False

    def close(self, reason: str = "close") -> dict[str, str]:
        if self._closed:
            return artifacts(self.output_dir)
        self._closed = True
        try:
            self.observer.close(reason)
        finally:
            self._clear()
        return artifacts(self.output_dir)


async def astart_observability(config: MiniSweAgentObservabilityConfig) -> MiniSweAgentRelayRun:
    """Initialize Relay observability and return a Mini SWE observer run."""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    await relay_plugin.initialize(
        relay_plugin.PluginConfig(components=[ComponentSpec(build_observability_config(config))])
    )
    observer = NemoRelayMiniSweObserver(
        instance_id=config.instance_id,
        model_name=config.model_name,
        task_index=config.task_index,
        rollout_index=config.rollout_index,
        extra=config.metadata(),
    )
    return MiniSweAgentRelayRun(observer=observer, output_dir=config.output_dir, _clear=relay_plugin.clear)


def start_observability(config: MiniSweAgentObservabilityConfig) -> MiniSweAgentRelayRun:
    """Synchronously initialize Relay observability and return a Mini SWE observer run."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        result: list[MiniSweAgentRelayRun] = []
        error: list[BaseException] = []

        def runner() -> None:
            try:
                result.append(asyncio.run(astart_observability(config)))
            except BaseException as exc:
                error.append(exc)

        thread = threading.Thread(target=runner, name="nemo-relay-mini-swe-observer", daemon=True)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]
    return asyncio.run(astart_observability(config))


__all__ = [
    "MiniSweAgentObservabilityConfig",
    "MiniSweAgentRelayRun",
    "NemoRelayMiniSweObserver",
    "artifacts",
    "astart_observability",
    "build_observability_config",
    "start_observability",
]
