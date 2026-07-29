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

from pathlib import Path

from nemo_rl.models.generation.sglang.utils import patches


def test_weight_update_session_completion_requires_cross_file_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    relative_paths = (
        "srt/managers/io_struct.py",
        "srt/entrypoints/http_server.py",
        "srt/managers/tokenizer_control_mixin.py",
        "srt/managers/scheduler.py",
        "srt/managers/scheduler_update_weights_mixin.py",
    )
    paths = {
        relative_path: tmp_path / relative_path.replace("/", "_")
        for relative_path in relative_paths
    }
    for path in paths.values():
        path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        patches,
        "_get_sglang_file",
        lambda relative_path: str(paths[relative_path]),
    )

    paths["srt/managers/io_struct.py"].write_text(
        "class BeginWeightUpdateReqInput:\n"
        "    pass\n"
        "class EndWeightUpdateReqInput:\n"
        "    pass\n",
        encoding="utf-8",
    )
    paths["srt/entrypoints/http_server.py"].write_text(
        '@app.post("/begin_weight_update")\n@app.post("/end_weight_update")\n',
        encoding="utf-8",
    )
    assert not patches._sglang_weight_update_session_is_complete()

    paths["srt/managers/tokenizer_control_mixin.py"].write_text(
        "async def begin_weight_update():\n"
        "    pass\n"
        "async def end_weight_update():\n"
        "    pass\n",
        encoding="utf-8",
    )
    paths["srt/managers/scheduler.py"].write_text(
        "self.weight_updater.begin_weight_update\n"
        "self.weight_updater.end_weight_update\n",
        encoding="utf-8",
    )
    assert patches._sglang_weight_update_session_is_complete()

    paths["srt/managers/scheduler.py"].write_text(
        "(BeginWeightUpdateReqInput, self.begin_weight_update)\n"
        "(EndWeightUpdateReqInput, self.end_weight_update)\n",
        encoding="utf-8",
    )
    paths["srt/managers/scheduler_update_weights_mixin.py"].write_text(
        "def begin_weight_update():\n"
        "    pass\n"
        "def end_weight_update():\n"
        "    pass\n"
        "update_weights_from_distributed requires an open session\n"
        "update_weights_from_tensor requires an open session\n",
        encoding="utf-8",
    )
    assert patches._sglang_weight_update_session_is_complete()
