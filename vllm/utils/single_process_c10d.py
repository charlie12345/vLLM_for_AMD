# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A single-rank stand-in for ``torch.distributed``.

Some PyTorch builds ship without the c10d extension. The AMD ROCm wheels for
native Windows are the motivating case: they are compiled with
``USE_DISTRIBUTED=0`` / ``USE_GLOO=OFF``, so ``torch._C._distributed_c10d``
does not exist, ``torch.distributed.is_available()`` is False, and importing
``torch.distributed._functional_collectives`` -- which ``parallel_state`` does
at module scope -- raises ``ModuleNotFoundError``. vLLM cannot be imported at
all on such a build.

vLLM only needs real collectives when a group spans more than one rank, and
``GroupCoordinator`` already short-circuits every collective it owns when
``world_size == 1``. This module fills in the gap for exactly that case: it
installs an in-process implementation where every group has a single rank, so
each collective is the identity. Anything that genuinely requires a peer raises
``RuntimeError`` rather than silently returning wrong data.

``install()`` is a no-op when torch has a real ``torch.distributed``.
"""

import sys
import types
from datetime import timedelta
from enum import Enum
from typing import Any

import torch

_WORLD_SIZE = 1
_RANK = 0


_installed = False


def is_needed() -> bool:
    """True when this torch build has no usable ``torch.distributed``.

    ``torch.distributed.is_available()`` is deliberately left reporting False
    even after install: torch's own internals (dynamo's trace rules, FSDP) key
    off it to decide whether to import distributed-only modules, and those
    really are unavailable. Only vLLM is meant to see the stand-in.
    """
    return not torch.distributed.is_available()


class ReduceOp(Enum):
    SUM = 0
    AVG = 1
    PRODUCT = 2
    MIN = 3
    MAX = 4
    BAND = 5
    BOR = 6
    BXOR = 7
    PREMUL_SUM = 8


class Backend(str):
    GLOO = "gloo"
    NCCL = "nccl"
    MPI = "mpi"
    UNDEFINED = "undefined"

    def __new__(cls, name: str):
        return super().__new__(cls, name.lower())


class Work:
    """A collective that has already completed, because there was nothing
    to exchange."""

    def wait(self, timeout: timedelta | None = None) -> bool:
        return True

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return True

    def get_future(self) -> Any:
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(None)
        return future


class ProcessGroup:
    """A group of exactly one rank."""

    class BackendType(Enum):
        UNDEFINED = 0
        GLOO = 1
        NCCL = 2
        CUSTOM = 3

    class Options:
        def __init__(self, backend: str = "gloo", timeout=None):
            self.backend = backend
            self._timeout = timeout

    def __init__(self, name: str = "default", backend: str = "gloo"):
        self.group_name = name
        self._backend = backend
        self.bound_device_id: torch.device | None = None

    def size(self) -> int:
        return _WORLD_SIZE

    def rank(self) -> int:
        return _RANK

    def name(self) -> str:
        return self.group_name

    def _get_backend(self, device: torch.device | None = None) -> "ProcessGroup":
        # vLLM only reaches for the backend object to poke at its options; the
        # group itself is a close enough stand-in for a single rank.
        return self

    def _get_backend_name(self) -> str:
        return self._backend

    def _set_default_backend(self, backend_type) -> None:
        pass

    def _register_backend(self, device, backend_type, backend_class) -> None:
        pass

    def _set_sequence_number_for_group(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<single-rank ProcessGroup {self.group_name!r} ({self._backend})>"


class ProcessGroupGloo(ProcessGroup):
    """What ``stateless_init_torch_distributed_process_group`` constructs."""

    def __init__(self, store=None, rank: int = 0, size: int = 1, timeout=None):
        if size != 1 or rank != 0:
            _no_peer(f"ProcessGroupGloo(rank={rank}, size={size})")
        super().__init__("gloo", "gloo")


class Store:
    """An in-process key/value store. Only one rank ever reads it."""

    def __init__(self, *args, **kwargs):
        self._data: dict[str, bytes] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value if isinstance(value, bytes) else str(value).encode()

    def get(self, key: str) -> bytes:
        if key not in self._data:
            raise RuntimeError(f"Key {key!r} not set in single-rank store")
        return self._data[key]

    def add(self, key: str, amount: int) -> int:
        total = int(self._data.get(key, b"0")) + amount
        self._data[key] = str(total).encode()
        return total

    def delete_key(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def wait(self, keys, timeout: timedelta | None = None) -> None:
        missing = [k for k in keys if k not in self._data]
        if missing:
            raise RuntimeError(f"Keys {missing} will never be set by a peer rank")

    def num_keys(self) -> int:
        return len(self._data)

    def set_timeout(self, timeout: timedelta) -> None:
        pass


class PrefixStore(Store):
    def __init__(self, prefix: str, store: Store | None = None, *args, **kwargs):
        super().__init__()
        self.prefix = prefix
        self.underlying_store = store


class DistError(RuntimeError):
    pass


class DistBackendError(DistError):
    pass


class DistNetworkError(DistError):
    pass


class DistStoreError(DistError):
    pass


_WORLD: ProcessGroup | None = None


class _GroupMember:
    WORLD: ProcessGroup | None = None
    NON_GROUP_MEMBER = -100


class _World:
    """Stands in for ``distributed_c10d._world``, the global group registry."""

    def __init__(self):
        self.pg_map: dict[ProcessGroup, Any] = {}
        self.pg_names: dict[ProcessGroup, str] = {}
        self.pg_group_ranks: dict[ProcessGroup, dict[int, int]] = {}
        self.pg_backend_config: dict[ProcessGroup, str] = {}
        self.group_count = 0

    @property
    def default_pg(self) -> ProcessGroup | None:
        return _WORLD

    @property
    def WORLD(self) -> ProcessGroup | None:
        return _WORLD


_world = _World()


def _get_default_timeout(backend=None) -> timedelta:
    return timedelta(minutes=10)


def _get_default_group() -> ProcessGroup:
    return _resolve(None)


def _register_process_group(group_name: str, pg: ProcessGroup) -> None:
    _world.pg_names[pg] = group_name


def _unregister_process_group(group_name: str) -> None:
    for pg, name in list(_world.pg_names.items()):
        if name == group_name:
            del _world.pg_names[pg]


def _resolve_process_group(group_name: str) -> ProcessGroup:
    for pg, name in _world.pg_names.items():
        if name == group_name:
            return pg
    raise RuntimeError(f"No process group registered as {group_name!r}")


def _no_peer(op: str):
    raise RuntimeError(
        f"torch.distributed.{op} needs at least one peer rank, but this "
        "PyTorch build has no c10d extension so vLLM is running against a "
        "single-rank stand-in. Only single-GPU (TP=PP=DP=1) execution is "
        "supported here."
    )


def _resolve(group) -> ProcessGroup:
    if group is None:
        if _WORLD is None:
            raise RuntimeError("Default process group has not been initialized")
        return _WORLD
    return group


# --- lifecycle -------------------------------------------------------------


def is_initialized() -> bool:
    return _WORLD is not None


def init_process_group(
    backend: str | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = "",
    pg_options: Any = None,
    device_id: torch.device | None = None,
    **kwargs,
) -> None:
    global _WORLD
    if world_size not in (-1, 1) or rank not in (-1, 0):
        _no_peer(f"init_process_group(world_size={world_size}, rank={rank})")
    if _WORLD is not None:
        raise RuntimeError("Default process group is already initialized")
    _WORLD = ProcessGroup("default", str(backend or "gloo"))
    _WORLD.bound_device_id = device_id
    _GroupMember.WORLD = _WORLD
    # Presence in pg_map is how callers tell a stateful group from one built
    # by vLLM's stateless_init_torch_distributed_process_group.
    _world.pg_map[_WORLD] = (str(backend or "gloo"), store)
    _world.pg_names[_WORLD] = "default"


def destroy_process_group(group: ProcessGroup | None = None) -> None:
    global _WORLD
    if group is None or group is _WORLD:
        _WORLD = None
        _GroupMember.WORLD = None


def new_group(
    ranks: list[int] | None = None,
    timeout: timedelta | None = None,
    backend: str | None = None,
    pg_options: Any = None,
    use_local_synchronization: bool = False,
    group_desc: str | None = None,
    device_id: torch.device | None = None,
    **kwargs,
) -> ProcessGroup:
    if ranks is not None and list(ranks) not in ([], [0]):
        _no_peer(f"new_group(ranks={list(ranks)})")
    _world.group_count += 1
    name = group_desc or f"group{_world.group_count}"
    group = ProcessGroup(name, str(backend or "gloo"))
    group.bound_device_id = device_id
    _world.pg_map[group] = (group._backend, None)
    _world.pg_names[group] = name
    return group


def split_group(
    parent_pg: ProcessGroup | None = None,
    split_ranks: list[list[int]] | None = None,
    timeout: timedelta | None = None,
    pg_options: Any = None,
    group_desc: str | None = None,
    **kwargs,
) -> ProcessGroup:
    return new_group(group_desc=group_desc, **kwargs)


def new_subgroups_by_enumeration(*args, **kwargs):
    return new_group(), [new_group()]


def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    yield Store(), 0, 1


# --- introspection ---------------------------------------------------------


def get_rank(group: ProcessGroup | None = None) -> int:
    return _RANK


def get_world_size(group: ProcessGroup | None = None) -> int:
    return _WORLD_SIZE


def get_backend(group: ProcessGroup | None = None) -> str:
    return _resolve(group)._get_backend_name()


def get_process_group_ranks(group: ProcessGroup | None = None) -> list[int]:
    return [_RANK]


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    if global_rank != _RANK:
        _no_peer(f"get_group_rank(global_rank={global_rank})")
    return _RANK


def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    if group_rank != _RANK:
        _no_peer(f"get_global_rank(group_rank={group_rank})")
    return _RANK


def is_backend_available(backend: str) -> bool:
    return str(backend).lower() == "gloo"


def is_gloo_available() -> bool:
    return True


def is_nccl_available() -> bool:
    return False


def is_mpi_available() -> bool:
    return False


def is_ucc_available() -> bool:
    return False


def is_xccl_available() -> bool:
    return False


def is_torchelastic_launched() -> bool:
    return False


def supports_complex(reduce_op: ReduceOp) -> bool:
    return True


# --- collectives (identity, because there is exactly one rank) -------------


def _done(async_op: bool):
    return Work() if async_op else None


def barrier(group=None, async_op=False, device_ids=None, **kwargs):
    return _done(async_op)


def monitored_barrier(group=None, timeout=None, wait_all_ranks=False):
    return None


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False, **kwargs):
    return _done(async_op)


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False, **kwargs):
    return _done(async_op)


def broadcast(tensor, src=0, group=None, async_op=False, **kwargs):
    return _done(async_op)


def all_gather(tensor_list, tensor, group=None, async_op=False, **kwargs):
    tensor_list[0].copy_(tensor)
    return _done(async_op)


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    output_tensor.copy_(input_tensor.reshape(output_tensor.shape))
    return _done(async_op)


def gather(tensor, gather_list=None, dst=0, group=None, async_op=False, **kwargs):
    if gather_list is not None:
        gather_list[0].copy_(tensor)
    return _done(async_op)


def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False, **kwargs):
    if scatter_list is not None:
        tensor.copy_(scatter_list[0])
    return _done(async_op)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    output.copy_(input_list[0])
    return _done(async_op)


def reduce_scatter_tensor(
    output, input, op=ReduceOp.SUM, group=None, async_op=False, **kwargs
):
    output.copy_(input.reshape(output.shape))
    return _done(async_op)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    output.copy_(input.reshape(output.shape))
    return _done(async_op)


def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    output_tensor_list[0].copy_(input_tensor_list[0])
    return _done(async_op)


def all_gather_object(object_list, obj, group=None):
    object_list[0] = obj


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    if object_gather_list is not None:
        object_gather_list[0] = obj


def scatter_object_list(
    scatter_object_output_list, scatter_object_input_list=None, src=0, group=None
):
    if scatter_object_input_list is not None:
        scatter_object_output_list[0] = scatter_object_input_list[0]


def broadcast_object_list(object_list, src=0, group=None, device=None):
    return None


# --- point-to-point (impossible with one rank) -----------------------------


def send(tensor, dst=None, group=None, tag=0, **kwargs):
    _no_peer("send")


def recv(tensor, src=None, group=None, tag=0, **kwargs):
    _no_peer("recv")


def isend(tensor, dst=None, group=None, tag=0, **kwargs):
    _no_peer("isend")


def irecv(tensor, src=None, group=None, tag=0, **kwargs):
    _no_peer("irecv")


def batch_isend_irecv(p2p_op_list):
    _no_peer("batch_isend_irecv")


class P2POp:
    def __init__(self, op, tensor, peer, group=None, tag=0):
        _no_peer("P2POp")


# --- submodule payloads ----------------------------------------------------


def _build_functional_collectives() -> types.ModuleType:
    """``torch.distributed._functional_collectives`` -- the async-tensor API.

    With one rank every collective is the identity and nothing is ever in
    flight, so the "async" tensors are just the inputs.
    """
    mod = types.ModuleType("torch.distributed._functional_collectives")

    def wait_tensor(tensor):
        return tensor

    def all_reduce_(tensor, reduceOp="sum", group=None, tag=""):
        return tensor

    def all_gather_tensor(tensor, gather_dim=0, group=None, tag=""):
        return tensor

    def reduce_scatter_tensor_(
        tensor, reduceOp="sum", scatter_dim=0, group=None, tag=""
    ):
        return tensor

    def all_to_all_single_(output, input_, *args, **kwargs):
        return output

    def broadcast_(tensor, src=0, group=None, tag=""):
        return tensor

    mod.wait_tensor = wait_tensor
    mod.all_reduce = all_reduce_
    mod.all_gather_tensor = all_gather_tensor
    mod.all_gather_into_tensor = all_gather_tensor
    mod.reduce_scatter_tensor = reduce_scatter_tensor_
    mod.all_to_all_single = all_to_all_single_
    mod.broadcast = broadcast_
    mod.AsyncCollectiveTensor = torch.Tensor
    return mod


def _build_symmetric_memory() -> types.ModuleType:
    """``torch.distributed._symmetric_memory`` -- NVLink/xGMI peer buffers.

    There is no peer to share memory with, so the module reports itself
    unavailable and vLLM takes its ordinary paths.
    """
    mod = types.ModuleType("torch.distributed._symmetric_memory")

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            "Symmetric memory is not available without torch.distributed"
        )

    mod.enable_symm_mem_for_group = unavailable
    mod.empty = unavailable
    mod.rendezvous = unavailable
    mod.is_nvshmem_available = lambda: False
    mod.get_symm_mem_workspace = unavailable
    mod._fused_scaled_matmul_reduce_scatter_impl = unavailable
    mod._fused_all_gather_matmul_impl = unavailable
    mod._pipelined_multi_all_gather_and_consume = unavailable
    return mod


def _build_distributed_c10d(dist: types.ModuleType) -> types.ModuleType:
    """``torch.distributed.distributed_c10d`` -- re-exports of the above.

    Callers reach into this module for the same names the package exposes, so
    it is a thin view over what we just installed.
    """
    mod = types.ModuleType("torch.distributed.distributed_c10d")
    for name in dir(dist):
        if not name.startswith("__"):
            setattr(mod, name, getattr(dist, name))
    mod.GroupMember = _GroupMember
    return mod


def _build_dtensor_modules() -> dict[str, types.ModuleType]:
    """``torch.distributed.tensor`` and friends -- the DTensor subsystem.

    vLLM never builds a DTensor, but transformers imports the API at module
    scope guarded only by ``is_torch_available()``, so the names have to
    resolve. They are deliberately inert: sharding a tensor across one rank is
    never something we should be asked to do, and doing it silently wrong would
    be worse than failing.
    """

    class Placement:
        def __init__(self, *args, **kwargs):
            pass

    class Shard(Placement):
        def __init__(self, dim: int = 0):
            self.dim = dim

        def local_shard_size_and_offset(self, *args, **kwargs):
            _no_peer("Shard.local_shard_size_and_offset")

    class Replicate(Placement):
        pass

    class Partial(Placement):
        def __init__(self, reduce_op: str = "sum"):
            self.reduce_op = reduce_op

    class DeviceMesh:
        def __init__(self, device_type: str = "cpu", mesh=None, **kwargs):
            self.device_type = device_type
            self.mesh = mesh

        def size(self, dim: int | None = None) -> int:
            return _WORLD_SIZE

        def get_rank(self) -> int:
            return _RANK

    class DTensor(torch.Tensor):
        """Nothing is ever an instance of this; it exists for isinstance()."""

        @staticmethod
        def from_local(*args, **kwargs):
            _no_peer("DTensor.from_local")

    def init_device_mesh(device_type, mesh_shape, **kwargs):
        if tuple(mesh_shape) not in ((), (1,)):
            _no_peer(f"init_device_mesh(mesh_shape={tuple(mesh_shape)})")
        return DeviceMesh(device_type, mesh_shape)

    def distribute_tensor(*args, **kwargs):
        _no_peer("distribute_tensor")

    def compute_local_shape_and_global_offset(*args, **kwargs):
        _no_peer("compute_local_shape_and_global_offset")

    tensor = types.ModuleType("torch.distributed.tensor")
    tensor.DTensor = DTensor
    tensor.DeviceMesh = DeviceMesh
    tensor.Shard = Shard
    tensor.Replicate = Replicate
    tensor.Partial = Partial
    tensor.Placement = Placement
    tensor.init_device_mesh = init_device_mesh
    tensor.distribute_tensor = distribute_tensor
    tensor.distribute_module = distribute_tensor
    tensor.zeros = distribute_tensor
    tensor.empty = distribute_tensor
    tensor.ones = distribute_tensor

    placement_types = types.ModuleType("torch.distributed.tensor.placement_types")
    placement_types.Placement = Placement
    placement_types.Shard = Shard
    placement_types.Replicate = Replicate
    placement_types.Partial = Partial
    tensor.placement_types = placement_types

    utils = types.ModuleType("torch.distributed.tensor._utils")
    utils.compute_local_shape_and_global_offset = compute_local_shape_and_global_offset
    tensor._utils = utils

    device_mesh = types.ModuleType("torch.distributed.device_mesh")
    device_mesh.DeviceMesh = DeviceMesh
    device_mesh.init_device_mesh = init_device_mesh

    return {
        "torch.distributed.tensor": tensor,
        "torch.distributed.tensor.placement_types": placement_types,
        "torch.distributed.tensor._utils": utils,
        "torch.distributed.device_mesh": device_mesh,
    }


def _build_fsdp_modules() -> dict[str, types.ModuleType]:
    """``torch.distributed.fsdp`` and ``._composable.fsdp`` -- sharded training.

    Inference never shards, but transformers imports the policy types at module
    scope behind a torch-version check rather than an availability check.
    """

    class _Policy:
        def __init__(self, *args, **kwargs):
            pass

    class FSDPModule:
        """dynamo's guard pickler issubclass()-tests against this; nothing
        here ever subclasses it."""

    def fully_shard(*args, **kwargs):
        _no_peer("fully_shard")

    fully_shard_mod = types.ModuleType("torch.distributed.fsdp._fully_shard")
    fully_shard_mod.FSDPModule = FSDPModule
    fully_shard_mod.fully_shard = fully_shard

    def _fully_shard_getattr(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # dynamo probes for _fsdp_param_group behind
        # `except ModuleNotFoundError`, and treats None as "no FSDP", which is
        # exactly right here. Reporting anything else makes it trace into a
        # subsystem that cannot work without real collectives.
        raise ModuleNotFoundError(
            f"torch.distributed.fsdp._fully_shard.{name} is not provided by "
            "vLLM's single-rank stand-in.",
            name="torch.distributed.fsdp._fully_shard",
        )

    fully_shard_mod.__getattr__ = _fully_shard_getattr

    fsdp = types.ModuleType("torch.distributed.fsdp")
    fsdp.__path__ = []  # marks it a package so submodule imports resolve
    fsdp.OffloadPolicy = _Policy
    fsdp.CPUOffloadPolicy = _Policy
    fsdp.MixedPrecisionPolicy = _Policy
    fsdp.FullyShardedDataParallel = _Policy
    fsdp.ShardingStrategy = _Policy
    fsdp.StateDictType = _Policy
    fsdp.FSDPModule = FSDPModule
    fsdp.fully_shard = fully_shard
    fsdp._fully_shard = fully_shard_mod

    composable = types.ModuleType("torch.distributed._composable")
    composable.__path__ = []
    composable_fsdp = types.ModuleType("torch.distributed._composable.fsdp")
    composable_fsdp.__path__ = []
    composable_fsdp.fully_shard = fully_shard
    composable_fsdp.FSDPModule = FSDPModule
    composable_fsdp.CPUOffloadPolicy = _Policy
    composable_fsdp.MixedPrecisionPolicy = _Policy
    composable_fsdp.OffloadPolicy = _Policy
    composable_fsdp._fully_shard = fully_shard_mod
    composable.fsdp = composable_fsdp

    return {
        "torch.distributed.fsdp": fsdp,
        "torch.distributed.fsdp._fully_shard": fully_shard_mod,
        "torch.distributed._composable": composable,
        "torch.distributed._composable.fsdp": composable_fsdp,
    }


def _build_c_extension() -> types.ModuleType:
    """``torch._C._distributed_c10d`` -- the pybind layer, in pure Python.

    Type annotations across vLLM reference it by that path, so it has to be
    both an attribute of ``torch._C`` and importable.
    """
    mod = types.ModuleType("torch._C._distributed_c10d")
    mod.ProcessGroup = ProcessGroup
    mod.ProcessGroupGloo = ProcessGroupGloo
    mod.Work = Work
    mod.ReduceOp = ReduceOp
    mod.Store = Store
    mod.PrefixStore = PrefixStore
    mod.TCPStore = TCPStore
    mod.FileStore = FileStore
    mod.HashStore = Store
    mod.BackendType = ProcessGroup.BackendType
    mod._DEFAULT_PG_TIMEOUT = _get_default_timeout()
    mod._register_process_group = _register_process_group
    mod._unregister_process_group = _unregister_process_group
    mod._resolve_process_group = _resolve_process_group

    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            # Introspection (inspect.getmodule probing __file__, pickling
            # probing __reduce__) must see a plain missing attribute.
            raise AttributeError(name)
        # Torch's own optional-import sites (dynamo's FSDP hooks, for one) are
        # written as `try: from torch.distributed... except ModuleNotFoundError`.
        # Anything we have not implemented genuinely is not there, so report it
        # the way an absent extension module would and let those guards fire.
        raise ModuleNotFoundError(
            f"torch._C._distributed_c10d.{name} is not provided by vLLM's "
            "single-rank stand-in; this PyTorch build has no c10d extension.",
            name="torch._C._distributed_c10d",
        )

    mod.__getattr__ = __getattr__
    return mod


def _build_rendezvous() -> types.ModuleType:
    """``torch.distributed.rendezvous`` -- how ranks find each other."""
    mod = types.ModuleType("torch.distributed.rendezvous")
    mod.rendezvous = rendezvous
    mod.register_rendezvous_handler = lambda scheme, handler: None
    return mod


_EXPORTS = [
    "ReduceOp",
    "Backend",
    "Work",
    "ProcessGroup",
    "ProcessGroupGloo",
    "Store",
    "PrefixStore",
    "TCPStore",
    "FileStore",
    "_world",
    "_get_default_timeout",
    "_get_default_group",
    "_register_process_group",
    "_unregister_process_group",
    "_resolve_process_group",
    "DistError",
    "DistBackendError",
    "DistNetworkError",
    "DistStoreError",
    "is_initialized",
    "init_process_group",
    "destroy_process_group",
    "new_group",
    "split_group",
    "new_subgroups_by_enumeration",
    "rendezvous",
    "get_rank",
    "get_world_size",
    "get_backend",
    "get_process_group_ranks",
    "get_group_rank",
    "get_global_rank",
    "is_backend_available",
    "is_gloo_available",
    "is_nccl_available",
    "is_mpi_available",
    "is_ucc_available",
    "is_xccl_available",
    "is_torchelastic_launched",
    "supports_complex",
    "barrier",
    "monitored_barrier",
    "all_reduce",
    "reduce",
    "broadcast",
    "all_gather",
    "all_gather_into_tensor",
    "gather",
    "scatter",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "all_to_all",
    "all_gather_object",
    "gather_object",
    "scatter_object_list",
    "broadcast_object_list",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "P2POp",
]

# Aliases for store types vLLM names but never needs a real implementation of.
TCPStore = Store
FileStore = Store


def install() -> bool:
    """Graft the stand-in onto ``torch.distributed``. Returns True if applied."""
    global _installed
    if _installed or not is_needed():
        return _installed
    _installed = True

    c_ext = _build_c_extension()
    sys.modules["torch._C._distributed_c10d"] = c_ext
    torch._C._distributed_c10d = c_ext

    dist = torch.distributed
    this = sys.modules[__name__]
    for name in _EXPORTS:
        setattr(dist, name, getattr(this, name))
    dist.GroupMember = _GroupMember
    dist.group = _GroupMember

    c10d = _build_distributed_c10d(dist)
    sys.modules["torch.distributed.distributed_c10d"] = c10d
    dist.distributed_c10d = c10d

    funcol = _build_functional_collectives()
    sys.modules["torch.distributed._functional_collectives"] = funcol
    dist._functional_collectives = funcol

    symm = _build_symmetric_memory()
    sys.modules["torch.distributed._symmetric_memory"] = symm
    dist._symmetric_memory = symm

    rdzv = _build_rendezvous()
    sys.modules["torch.distributed.rendezvous"] = rdzv
    dist.rendezvous = rdzv.rendezvous

    for name, mod in (_build_dtensor_modules() | _build_fsdp_modules()).items():
        sys.modules[name] = mod
        parent, _, leaf = name.rpartition(".")
        # Only bind direct children onto the package, or torch.distributed.fsdp
        # would be clobbered by torch.distributed._composable.fsdp.
        if parent == "torch.distributed":
            setattr(dist, leaf, mod)
    dist.DeviceMesh = sys.modules["torch.distributed.device_mesh"].DeviceMesh
    dist.init_device_mesh = sys.modules[
        "torch.distributed.device_mesh"
    ].init_device_mesh

    return True
