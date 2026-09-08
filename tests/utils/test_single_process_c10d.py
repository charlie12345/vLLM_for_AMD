# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch

from vllm.utils import single_process_c10d as c10d


def test_functional_all_to_all_single_copies_input() -> None:
    functional = c10d._build_functional_collectives()
    input_tensor = torch.arange(4)
    output_tensor = torch.empty((2, 2), dtype=input_tensor.dtype)

    result = functional.all_to_all_single(output_tensor, input_tensor)

    assert result is output_tensor
    torch.testing.assert_close(output_tensor, input_tensor.reshape(2, 2))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: c10d.reduce(torch.zeros(1), dst=1),
        lambda: c10d.broadcast(torch.zeros(1), src=1),
        lambda: c10d.gather(torch.zeros(1), dst=1),
        lambda: c10d.scatter(torch.zeros(1), src=1),
        lambda: c10d.gather_object(object(), dst=1),
        lambda: c10d.scatter_object_list([None], src=1),
        lambda: c10d.broadcast_object_list([None], src=1),
        lambda: list(c10d.rendezvous("env://", rank=1, world_size=2)),
        lambda: c10d._build_functional_collectives().broadcast(torch.zeros(1), src=1),
    ],
)
def test_collectives_reject_peer_rank(operation: Callable[[], object]) -> None:
    with pytest.raises(RuntimeError, match="needs at least one peer rank"):
        operation()
