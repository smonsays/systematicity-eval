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

"""Configuration schemas."""

import dataclasses


def load_prompt(prompt_name: str) -> str:
  """Load a prompt from the prompts directory."""
  with open(f'prompts/{prompt_name}.md', 'r', encoding='utf-8') as f:
    return f.read().strip()


@dataclasses.dataclass
class BooleanDatasetConfig:
  name: str
  n_features: int
  feature_maxval: int
  n_query: int
  min_parse_tree_size: int
  max_parse_tree_size: int
  n_rules_per_size: int
  n_translations_per_rule: int
  n_shuffle_per_rule: int


@dataclasses.dataclass
class RavenDatasetConfig:
  name: str
  n_sets: int
  n_permutations_per_set: int
  n_features: int
  feature_maxval: int


@dataclasses.dataclass
class MLCDatasetConfig:
  name: str
  vocabulary: str
  n_languages: int
  n_translation_per_language: int
  n_recomposition_per_language: int
  n_shuffle_per_language: int
  n_support: int
  n_query: int
  n_primitives: int
  n_rules: int
  max_rhs_length: int
  max_example_length: int
  p_lhs_onearg: float


@dataclasses.dataclass
class ListintDatasetConfig:
  name: str
  n_support: int
  n_query: int
  min_parse_tree_size: int
  max_parse_tree_size: int
  n_rules_per_size: int
  n_translation_per_rule: int
  n_shuffle_per_rule: int
  list_maxlength: int
  feature_maxval: int
  easy_instructions: bool


@dataclasses.dataclass
class ModelConfig:
  name: str
  provider: str | None
  max_concurrent: int
  reasoning_effort: str
  temperature: float | None
  explain: bool
  n_attempts: int


@dataclasses.dataclass
class Config:
  dataset: (
    BooleanDatasetConfig | RavenDatasetConfig | MLCDatasetConfig | ListintDatasetConfig
  )
  model: ModelConfig
  seed: int
  logger: str | None = None
