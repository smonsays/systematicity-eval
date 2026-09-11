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

"""Configuration for raven experiments."""

from synthsys import config_classes


def get_config() -> config_classes.Config:
  dataset = config_classes.RavenDatasetConfig(
    name='sraven',
    n_sets=10,
    n_permutations_per_set=5,
    n_features=3,
    feature_maxval=10,
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
