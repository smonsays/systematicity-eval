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

"""Configuration for Boolean rule induction experiments."""

from synthsys import config_classes


def get_config() -> config_classes.Config:
  dataset = config_classes.BooleanDatasetConfig(
    name='boolean',
    n_features=3,
    feature_maxval=3,
    n_query=8,
    min_parse_tree_size=16,
    max_parse_tree_size=32,
    n_rules_per_size=3,
    n_translations_per_rule=4,
    n_shuffle_per_rule=4,
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
