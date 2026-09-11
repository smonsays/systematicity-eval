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

"""Configuration for mlc experiments."""

from synthsys import config_classes


def get_config() -> config_classes.Config:
  dataset = config_classes.MLCDatasetConfig(
    name='mlc',
    vocabulary='MLC',
    n_languages=10,
    n_translation_per_language=4,
    n_recomposition_per_language=4,
    n_shuffle_per_language=4,
    n_support=4,
    n_query=10,
    n_primitives=4,
    n_rules=3,
    max_rhs_length=3,
    max_example_length=30,
    p_lhs_onearg=0.5,
  )

  model = config_classes.ModelConfig(
    name='gemini-3.1-flash-lite-preview',
    provider='gemini',
    max_concurrent=5,
    reasoning_effort='auto',
    temperature=None,
    explain=False,
    n_attempts=1,
  )

  config = config_classes.Config(
    seed=0,
    dataset=dataset,
    model=model,
    logger=None,
  )

  return config
