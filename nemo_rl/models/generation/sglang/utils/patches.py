# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def _get_sglang_file(relative_path: str) -> str:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"sglang package not found while attempting to patch '{relative_path}'. "
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected sglang file '{relative_path}' not found at '{file_path}'. "
            "The sglang version may have moved this file; compat patch cannot be applied."
        )
    return file_path


def _write_and_verify(
    file_path: str, content: str, sentinel: str | tuple[str, ...]
) -> None:
    sentinels = (sentinel,) if isinstance(sentinel, str) else sentinel
    tmp_path = f"{file_path}.nemo_rl_compat.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    with open(file_path, "r") as f:
        verify = f.read()
    missing = [item for item in sentinels if item not in verify]
    if missing:
        raise RuntimeError(
            f"Compat patch verification failed for {file_path}: "
            f"sentinel(s) {missing} not present after write. "
            "The write may have been silently dropped by the filesystem."
        )


def _patch_sglang_file_replacements(
    relative_path: str,
    replacements: tuple[tuple[str, str, str], ...],
    description: str,
) -> None:
    file_to_patch = _get_sglang_file(relative_path)

    with open(file_to_patch, "r") as f:
        content = f.read()

    missing_replacements = [
        (sentinel, anchor, replacement)
        for sentinel, anchor, replacement in replacements
        if sentinel not in content
    ]
    if not missing_replacements:
        return

    for sentinel, anchor, replacement in missing_replacements:
        if anchor not in content:
            raise RuntimeError(
                f"{description} anchor for sentinel '{sentinel}' not found in "
                f"{file_to_patch}."
            )
        content = content.replace(anchor, replacement, 1)

    _write_and_verify(
        file_to_patch, content, tuple(sentinel for sentinel, _, _ in replacements)
    )
    logger.info("Patched %s in %s.", description, file_to_patch)


def _patch_sglang_safe_unpickler() -> None:
    file_to_patch = _get_sglang_file("srt/utils/common.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = '"nemo_rl.models.generation.sglang.utils.train_utils."'
    if sentinel in content:
        return

    anchor = '        "torch.nn.parameter.",\n'
    insertion = (
        anchor + '        "nemo_rl.models.generation.sglang.utils.train_utils.",\n'
    )
    if anchor not in content:
        raise RuntimeError(
            f"SafeUnpickler allowlist anchor '{anchor.strip()}' not found in "
            f"{file_to_patch}."
        )

    content = content.replace(anchor, insertion, 1)
    _write_and_verify(file_to_patch, content, sentinel)
    logger.info("Patched SafeUnpickler allowlist in %s.", file_to_patch)


def _override_sglang_imbalance_check_env() -> None:
    """Force-disable sglang's per-GPU memory imbalance check.

    Pop the legacy names so the shim has nothing to copy, then set
    ``ENABLE=false`` directly. Inherited env reaches the subprocesses
    cleaned, so the shim no longer overwrites our ENABLE on re-import.
    """
    for legacy in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        os.environ.pop(legacy, None)
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"


def _get_megatron_file(subpackage: str, relative_path: str) -> str | None:
    """Locate a file inside ``megatron.<subpackage>`` (e.g. ``core``, ``training``).

    Returns ``None`` if megatron isn't importable so callers can treat that
    as "nothing to patch". Raises if the package is present but the
    expected file is missing (signals a megatron version mismatch).
    """
    full_pkg = f"megatron.{subpackage}"
    try:
        spec = find_spec(full_pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected megatron file '{full_pkg}/{relative_path}' not found at "
            f"'{file_path}'. The megatron version may have moved this file; "
            "compat patch cannot be applied."
        )
    return file_path


def _patch_megatron_hook_mode_in(file_path: str) -> None:
    """Comment out ``torch_memory_saver.hook_mode = "torch"`` in a megatron file.

    Megatron sets ``tms.hook_mode = "torch"`` at module import time on the
    global ``torch_memory_saver`` singleton. That mutation breaks sglang's
    pauseable CUDA graph path, which asserts ``_hook_mode == "preload"``
    inside ``TorchMemorySaver.cuda_graph(...)``. Commenting the line out
    leaves the singleton at its default ``"preload"`` mode that sglang
    expects.
    """
    with open(file_path, "r") as f:
        content = f.read()

    sentinel = '# torch_memory_saver.hook_mode = "torch"'
    if sentinel in content:
        return

    anchor = '    torch_memory_saver.hook_mode = "torch"\n'
    if anchor not in content:
        raise RuntimeError(
            f"Megatron hook_mode anchor '{anchor.strip()}' not found in "
            f"{file_path}; the megatron version may have moved or removed it."
        )

    replacement = (
        '    # torch_memory_saver.hook_mode = "torch"  '
        "# patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n"
    )
    content = content.replace(anchor, replacement, 1)
    _write_and_verify(file_path, content, sentinel)
    logger.info("Patched megatron tms.hook_mode mutation in %s.", file_path)


def _patch_megatron_dynamic_context_hook_mode() -> None:
    file_path = _get_megatron_file("core", "inference/contexts/dynamic_context.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_megatron_training_hook_mode() -> None:
    file_path = _get_megatron_file("training", "training.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_sglang_custom_all_reduce_v2_tms_cudagraph() -> None:
    """Backport sglang#27948 for colocated TMS CUDA graph capture.

    With ``SGLANG_MEMORY_SAVER_CUDA_GRAPH=true``, custom all-reduce v2 must
    not flag the kernel as capturing. TMS replaces the IPC addresses during
    capture, so the addresses registered while ``set_cuda_graph_capture`` is
    on become stale and custom_all_reduce.cuh can fail at replay time. Passing
    ``not self.tms_cudagraph`` keeps the capture flag off in that mode.
    """
    _patch_sglang_file_replacements(
        "srt/distributed/device_communicators/custom_all_reduce_v2.py",
        (
            (
                "from sglang.srt.environ import envs\n",
                "from sglang.srt.utils import is_sm100_supported, log_info_on_rank0\n",
                "from sglang.srt.environ import envs\n"
                "from sglang.srt.utils import is_sm100_supported, log_info_on_rank0\n",
            ),
            (
                "        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n",
                "        self.override_algo: Optional[AllReduceAlgo] = None\n"
                "        self.obj = get_custom_all_reduce_cls()(\n",
                "        self.override_algo: Optional[AllReduceAlgo] = None\n"
                "        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()\n"
                "        self.obj = get_custom_all_reduce_cls()(\n",
            ),
            (
                "            self.obj.set_cuda_graph_capture(not self.tms_cudagraph)\n",
                "        try:\n            self.obj.set_cuda_graph_capture(True)\n",
                "        try:\n"
                "            self.obj.set_cuda_graph_capture(not self.tms_cudagraph)\n",
            ),
            (
                "                self.obj.set_cuda_graph_capture(not self.tms_cudagraph)\n",
                "            finally:\n"
                "                self.obj.set_cuda_graph_capture(True)\n",
                "            finally:\n"
                "                self.obj.set_cuda_graph_capture(not self.tms_cudagraph)\n",
            ),
        ),
        "custom all-reduce v2 TMS CUDA graph capture path",
    )


def _sglang_weight_update_session_is_complete() -> bool:
    """Return whether every component of a native or backported session exists."""

    def has_all(relative_path: str, sentinels: tuple[str, ...]) -> bool:
        with open(_get_sglang_file(relative_path)) as file:
            content = file.read()
        return all(sentinel in content for sentinel in sentinels)

    common_session_components = (
        has_all(
            "srt/managers/io_struct.py",
            (
                "class BeginWeightUpdateReqInput",
                "class EndWeightUpdateReqInput",
            ),
        )
        and has_all(
            "srt/entrypoints/http_server.py",
            (
                '@app.post("/begin_weight_update")',
                '@app.post("/end_weight_update")',
            ),
        )
        and has_all(
            "srt/managers/tokenizer_control_mixin.py",
            (
                "async def begin_weight_update(",
                "async def end_weight_update(",
            ),
        )
    )
    if not common_session_components:
        return False

    # sglang-miles routes lifecycle requests through ``weight_updater``.
    native_session = has_all(
        "srt/managers/scheduler.py",
        (
            "self.weight_updater.begin_weight_update",
            "self.weight_updater.end_weight_update",
        ),
    )
    # The v0.5.12.post1 backport dispatches directly to methods added to
    # SchedulerUpdateWeightsMixin.
    backported_session = has_all(
        "srt/managers/scheduler.py",
        (
            "(BeginWeightUpdateReqInput, self.begin_weight_update)",
            "(EndWeightUpdateReqInput, self.end_weight_update)",
        ),
    ) and has_all(
        "srt/managers/scheduler_update_weights_mixin.py",
        (
            "def begin_weight_update(",
            "def end_weight_update(",
            "update_weights_from_distributed requires an open session",
            "update_weights_from_tensor requires an open session",
        ),
    )
    return native_session or backported_session


def _patch_sglang_weight_update_session() -> None:
    """Backport SGLang's begin/end refit session API.

    The pinned release loads tensors but has no lifecycle hook to rerun
    quantization post-processing once all buckets arrive, so quantized
    weights keep stale derived scales.
    """
    # The patch spans five files, so only a complete implementation counts as
    # already-patched: a single endpoint may mean a concurrent Ray actor is
    # still mid-patch, or died partway through.
    if _sglang_weight_update_session_is_complete():
        return

    _patch_sglang_file_replacements(
        "srt/managers/io_struct.py",
        (
            (
                "class BeginWeightUpdateReqInput(BaseReq):",
                "@dataclass\nclass CheckWeightsReqInput(BaseReq):\n",
                "@dataclass\n"
                "class BeginWeightUpdateReqInput(BaseReq):\n"
                "    pass\n\n\n"
                "@dataclass\n"
                "class BeginWeightUpdateReqOutput(BaseReq):\n"
                "    success: bool\n"
                "    message: str\n\n\n"
                "@dataclass\n"
                "class EndWeightUpdateReqInput(BaseReq):\n"
                "    pass\n\n\n"
                "@dataclass\n"
                "class EndWeightUpdateReqOutput(BaseReq):\n"
                "    success: bool\n"
                "    message: str\n\n\n"
                "@dataclass\n"
                "class CheckWeightsReqInput(BaseReq):\n",
            ),
        ),
        "weight-update session request types",
    )
    _patch_sglang_file_replacements(
        "srt/entrypoints/http_server.py",
        (
            (
                "    BeginWeightUpdateReqInput,\n",
                "    AttachHiCacheStorageReqInput,\n",
                "    AttachHiCacheStorageReqInput,\n    BeginWeightUpdateReqInput,\n",
            ),
            (
                "    EndWeightUpdateReqInput,\n",
                "    DumperControlReqInput,\n",
                "    DumperControlReqInput,\n    EndWeightUpdateReqInput,\n",
            ),
            (
                '@app.post("/begin_weight_update")',
                '@app.post("/update_weights_from_tensor")\n',
                '@app.post("/begin_weight_update")\n'
                "@auth_level(AuthLevel.ADMIN_OPTIONAL)\n"
                "async def begin_weight_update(\n"
                "    obj: BeginWeightUpdateReqInput, request: Request\n"
                "):\n"
                '    """Open a weight-update session before loading buckets."""\n'
                "    success, message = (\n"
                "        await _global_state.tokenizer_manager.begin_weight_update(\n"
                "            obj, request\n"
                "        )\n"
                "    )\n"
                '    content = {"success": success, "message": message}\n'
                "    return ORJSONResponse(\n"
                "        content,\n"
                "        status_code=200 if success else HTTPStatus.BAD_REQUEST,\n"
                "    )\n\n\n"
                '@app.post("/end_weight_update")\n'
                "@auth_level(AuthLevel.ADMIN_OPTIONAL)\n"
                "async def end_weight_update(\n"
                "    obj: EndWeightUpdateReqInput, request: Request\n"
                "):\n"
                '    """Finalize quantized weights after all buckets arrive."""\n'
                "    success, message = (\n"
                "        await _global_state.tokenizer_manager.end_weight_update(\n"
                "            obj, request\n"
                "        )\n"
                "    )\n"
                '    content = {"success": success, "message": message}\n'
                "    return ORJSONResponse(\n"
                "        content,\n"
                "        status_code=200 if success else HTTPStatus.BAD_REQUEST,\n"
                "    )\n\n\n"
                '@app.post("/update_weights_from_tensor")\n',
            ),
        ),
        "weight-update session HTTP endpoints",
    )
    _patch_sglang_file_replacements(
        "srt/managers/tokenizer_control_mixin.py",
        (
            (
                "    BeginWeightUpdateReqInput,\n",
                "    AttachHiCacheStorageReqOutput,\n",
                "    AttachHiCacheStorageReqOutput,\n"
                "    BeginWeightUpdateReqInput,\n"
                "    BeginWeightUpdateReqOutput,\n",
            ),
            (
                "    EndWeightUpdateReqInput,\n",
                "    DumperControlReqOutput,\n",
                "    DumperControlReqOutput,\n"
                "    EndWeightUpdateReqInput,\n"
                "    EndWeightUpdateReqOutput,\n",
            ),
            (
                '    ("begin_weight_update", BeginWeightUpdateReqOutput),\n',
                '    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),\n',
                '    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),\n'
                '    ("begin_weight_update", BeginWeightUpdateReqOutput),\n'
                '    ("end_weight_update", EndWeightUpdateReqOutput),\n',
            ),
            (
                "    async def _nemo_rl_weight_update_session_call(",
                "    async def update_weights_from_distributed(\n",
                "    async def _nemo_rl_weight_update_session_call(\n"
                "        self: TokenizerManager, communicator, obj\n"
                "    ) -> Tuple[bool, str]:\n"
                '        """Run a refit lifecycle RPC with pause-aware locking."""\n'
                "        self.auto_create_handle_loop()\n"
                "        async with self.is_pause_cond:\n"
                "            is_paused = self.is_pause\n"
                "            if is_paused:\n"
                "                results = await communicator(obj)\n"
                "        if not is_paused:\n"
                "            async with self.model_update_lock.writer_lock:\n"
                "                results = await communicator(obj)\n"
                "        return FanOutCommunicator.merge_results(results)\n\n"
                "    async def begin_weight_update(\n"
                "        self: TokenizerManager,\n"
                "        obj: BeginWeightUpdateReqInput,\n"
                "        request: Optional[fastapi.Request] = None,\n"
                "    ) -> Tuple[bool, str]:\n"
                "        return await self._nemo_rl_weight_update_session_call(\n"
                "            self.begin_weight_update_communicator, obj\n"
                "        )\n\n"
                "    async def end_weight_update(\n"
                "        self: TokenizerManager,\n"
                "        obj: EndWeightUpdateReqInput,\n"
                "        request: Optional[fastapi.Request] = None,\n"
                "    ) -> Tuple[bool, str]:\n"
                "        return await self._nemo_rl_weight_update_session_call(\n"
                "            self.end_weight_update_communicator, obj\n"
                "        )\n\n"
                "    async def update_weights_from_distributed(\n",
            ),
        ),
        "weight-update session tokenizer fanout",
    )
    _patch_sglang_file_replacements(
        "srt/managers/scheduler.py",
        (
            (
                "    BeginWeightUpdateReqInput,\n",
                "    AttachHiCacheStorageReqOutput,\n",
                "    AttachHiCacheStorageReqOutput,\n    BeginWeightUpdateReqInput,\n",
            ),
            (
                "    EndWeightUpdateReqInput,\n",
                "    DumperControlReqOutput,\n",
                "    DumperControlReqOutput,\n    EndWeightUpdateReqInput,\n",
            ),
            (
                "                (BeginWeightUpdateReqInput, self.begin_weight_update),\n",
                "                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),\n",
                "                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),\n"
                "                (BeginWeightUpdateReqInput, self.begin_weight_update),\n"
                "                (EndWeightUpdateReqInput, self.end_weight_update),\n",
            ),
        ),
        "weight-update session scheduler dispatch",
    )
    _patch_sglang_file_replacements(
        "srt/managers/scheduler_update_weights_mixin.py",
        (
            (
                "    BeginWeightUpdateReqInput,\n",
                "    CheckWeightsReqInput,\n",
                "    BeginWeightUpdateReqInput,\n"
                "    BeginWeightUpdateReqOutput,\n"
                "    CheckWeightsReqInput,\n",
            ),
            (
                "    EndWeightUpdateReqInput,\n",
                "    GetWeightsByNameReqInput,\n",
                "    EndWeightUpdateReqInput,\n"
                "    EndWeightUpdateReqOutput,\n"
                "    GetWeightsByNameReqInput,\n",
            ),
            (
                "def _nemo_rl_run_quant_method_hook(",
                "logger = logging.getLogger(__name__)\n\n\n"
                "class SchedulerUpdateWeightsMixin:\n",
                "logger = logging.getLogger(__name__)\n\n\n"
                "def _nemo_rl_run_quant_method_hook(model, target_device, hook_name):\n"
                "    from sglang.srt.lora.layers import BaseLayerWithLoRA\n"
                "    from sglang.srt.model_loader.loader import device_loading_context\n\n"
                "    for _, module in model.named_modules():\n"
                "        if isinstance(module, BaseLayerWithLoRA):\n"
                "            continue\n"
                '        quant_method = getattr(module, "quant_method", None)\n'
                "        if quant_method is not None and hasattr(quant_method, hook_name):\n"
                "            with device_loading_context(module, target_device):\n"
                "                getattr(quant_method, hook_name)(module)\n\n\n"
                "class SchedulerUpdateWeightsMixin:\n",
            ),
            (
                "    def begin_weight_update(",
                "    def update_weights_from_distributed(\n",
                "    def begin_weight_update(\n"
                "        self: Scheduler, recv_req: BeginWeightUpdateReqInput\n"
                "    ):\n"
                '        assert not getattr(self, "_weight_update_in_progress", False), (\n'
                '            "begin_weight_update called while a session is already open"\n'
                "        )\n"
                "        runner = self.tp_worker.model_runner\n"
                "        _nemo_rl_run_quant_method_hook(\n"
                '            runner.model, torch.device(runner.device), "restore_weights_before_loading"\n'
                "        )\n"
                "        self._weight_update_in_progress = True\n"
                "        torch.distributed.barrier(group=self.tp_cpu_group)\n"
                '        return BeginWeightUpdateReqOutput(True, "Success")\n\n'
                "    def end_weight_update(\n"
                "        self: Scheduler, recv_req: EndWeightUpdateReqInput\n"
                "    ):\n"
                '        assert getattr(self, "_weight_update_in_progress", False), (\n'
                '            "end_weight_update called without begin_weight_update"\n'
                "        )\n"
                "        runner = self.tp_worker.model_runner\n"
                "        _nemo_rl_run_quant_method_hook(\n"
                '            runner.model, torch.device(runner.device), "process_weights_after_loading"\n'
                "        )\n"
                "        self._weight_update_in_progress = False\n"
                "        torch.distributed.barrier(group=self.tp_cpu_group)\n"
                '        return EndWeightUpdateReqOutput(True, "Success")\n\n'
                "    def update_weights_from_distributed(\n",
            ),
            (
                '        assert getattr(self, "_weight_update_in_progress", False), (\n'
                '            "update_weights_from_distributed requires an open session"\n'
                "        )\n",
                '        """Update the online model parameter."""\n'
                "        success, message = self.tp_worker.update_weights_from_distributed(recv_req)\n",
                '        """Update the online model parameter."""\n'
                '        assert getattr(self, "_weight_update_in_progress", False), (\n'
                '            "update_weights_from_distributed requires an open session"\n'
                "        )\n"
                "        success, message = self.tp_worker.update_weights_from_distributed(recv_req)\n",
            ),
            (
                '        assert getattr(self, "_weight_update_in_progress", False), (\n'
                '            "update_weights_from_tensor requires an open session"\n'
                "        )\n",
                '        """Update the online model parameter from tensors."""\n'
                "        if recv_req.disable_draft_model:\n",
                '        """Update the online model parameter from tensors."""\n'
                '        assert getattr(self, "_weight_update_in_progress", False), (\n'
                '            "update_weights_from_tensor requires an open session"\n'
                "        )\n"
                "        if recv_req.disable_draft_model:\n",
            ),
        ),
        "quantized weight-update session lifecycle",
    )
    if not _sglang_weight_update_session_is_complete():
        raise RuntimeError(
            "SGLang weight-update session patch did not produce a complete "
            "cross-file implementation."
        )


def _apply_sglang_compat_patches() -> None:
    _patch_sglang_safe_unpickler()
    _patch_sglang_custom_all_reduce_v2_tms_cudagraph()
    _patch_sglang_weight_update_session()
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()
