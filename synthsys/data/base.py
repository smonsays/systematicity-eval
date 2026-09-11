# Copyright 2026 Simon Schug
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

"""Base data structures for datasets."""

import jaxtyping as jt
from flax import struct


@struct.dataclass
class Batch:
  """
  A `Batch` contains a single problem instance to be presented to an LLM.
  The `target` can have multiple features and each of these features might
  be governed by a separate`rule`.
  As a result, all jt.Array are expected to have the same shape as `target`
  and must be duplicated where required.
  """

  prompt: str
  target: jt.Shaped[jt.ArrayLike, ' feature']
  rule: jt.Shaped[jt.ArrayLike, ' feature']
  info: dict[str, jt.Shaped[jt.ArrayLike, ' feature']] = struct.field(
    default_factory=dict
  )
