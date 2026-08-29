# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import vllm.v1.engine.utils as engine_utils


class _FakeProcess:
    name = "EngineCore"

    def __init__(self) -> None:
        self.join_timeout: float | None = None

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout


def test_cooperative_shutdown_waits_before_forced_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = Mock()
    process = _FakeProcess()
    forced_shutdown = Mock()
    monkeypatch.setattr(engine_utils, "shutdown", forced_shutdown)
    monkeypatch.setattr(
        engine_utils.envs,
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
        7,
    )
    monkeypatch.setattr(engine_utils.time, "monotonic", lambda: 100.0)

    engine_utils._shutdown_core_processes(
        [process],  # type: ignore[list-item]
        shutdown_event=event,
        timeout=3.0,
    )

    event.set.assert_called_once_with()
    assert process.join_timeout == 17.0
    forced_shutdown.assert_called_once_with([process], timeout=3.0)


def test_posix_shutdown_path_remains_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()
    forced_shutdown = Mock()
    monkeypatch.setattr(engine_utils, "shutdown", forced_shutdown)

    engine_utils._shutdown_core_processes([process], timeout=2.0)  # type: ignore[list-item]

    assert process.join_timeout is None
    forced_shutdown.assert_called_once_with([process], timeout=2.0)
