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

from collections import Counter
from typing import Literal

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from synthsys.data import boolean
from synthsys.model.api import APIModel


class BooleanTestCase(parameterized.TestCase):
  def test_concept_generator_sample(self) -> None:
    n_features = 3
    feature_maxval = 3
    n_support = 8
    n_query = 16
    generator = boolean.BooleanConceptGenerator(
      n_features=n_features, feature_maxval=feature_maxval, seed=0
    )
    task = generator.sample(n_support=n_support, n_query=n_query)

    self.assertEqual(task.inputs_support.shape, (n_support, n_features))
    self.assertEqual(task.outputs_support.shape, (n_support,))
    self.assertEqual(task.inputs_query.shape, (n_query, n_features))
    self.assertEqual(task.outputs_query.shape, (n_query,))
    self.assertEqual(task.output_table.shape, (feature_maxval**n_features,))

  def test_concept_generator_sample_curriculum(self) -> None:
    n_features = 3
    feature_maxval = 3
    n_query = 8
    generator = boolean.BooleanConceptGenerator(
      n_features=n_features, feature_maxval=feature_maxval, seed=0
    )
    task = generator.sample_curriculum(n_query=n_query)
    ideal_observer = boolean.BooleanIdealObserver(
      n_features=n_features,
      feature_maxval=feature_maxval,
      max_parse_tree_size=len(task.parse_tree),
      seed=0,
    )
    n_support = len(np.unique(task.inputs_support, axis=0))
    n_query = len(np.unique(task.inputs_query, axis=0))
    print('n_support: ', n_support, ' n_query: ', n_query)
    self.assertLessEqual(n_support + n_query, len(task.output_table))
    self.assertEqual(task.inputs_support.shape, (n_support, n_features))
    self.assertEqual(task.outputs_support.shape, (n_support,))
    self.assertEqual(task.inputs_query.shape, (n_query, n_features))
    self.assertEqual(task.outputs_query.shape, (n_query,))
    self.assertEqual(task.output_table.shape, (feature_maxval**n_features,))

    # The simplest rule that explains the support set should yield the same output
    # table as the rule that generated the task, otherwise the support set is not minimal
    prediction = ideal_observer.maximum_aposteriori(
      task.inputs_support, task.outputs_support, task.inputs_query
    )
    np.testing.assert_array_equal(prediction.targets, task.outputs_query)
    self.assertTrue(prediction.is_unique, 'The simplest rule should be unique')

  @parameterized.parameters(
    dict(n_features=3, feature_maxval=3),
    # dict(n_features=5, feature_maxval=3),
    # dict(n_features=4, feature_maxval=4),
    # dict(n_features=5, feature_maxval=5),
  )
  def test_create_dataloader(self, n_features: int, feature_maxval: int) -> None:
    n_query = 8
    n_rules_per_size = 10
    n_translations_per_rule = 0
    n_shuffle_per_rule = 0
    min_parse_tree_size = 16
    max_parse_tree_size = 32

    dataloader, _, _ = boolean.create_boolean_dataloader(
      n_features=n_features,
      feature_maxval=feature_maxval,
      n_query=n_query,
      n_shuffle_per_rule=n_shuffle_per_rule,
      min_parse_tree_size=min_parse_tree_size,
      max_parse_tree_size=max_parse_tree_size,
      n_rules_per_size=n_rules_per_size,
      n_translations_per_rule=n_translations_per_rule,
      seed=0,
    )

    batches = list(dataloader)
    self.assertGreater(len(batches), 0)

    # Dataloader skips ambiguous rules
    max_rules = (max_parse_tree_size - min_parse_tree_size + 1) * n_rules_per_size
    self.assertLessEqual(len(batches), max_rules)

    # self.assertEqual(batches[0].target.shape, (n_query,))
    # self.assertEqual(batches[0].rule.shape, (n_query,))

    n_batches_per_rule = 1 + n_translations_per_rule + n_shuffle_per_rule
    self.assertEqual(len(batches) % n_batches_per_rule, 0)

    sizes = [int(batch.info['parse_tree_size'][0]) for batch in batches]
    size_counts = Counter(sizes)

    print(f'{len(batches)} tasks contained in dataloader.')
    print(f'Tasks per parse tree size: {dict(sorted(size_counts.items()))}')

  @absltest.skip('Skipping long-running sample response test')
  @parameterized.parameters(
    dict(
      n_query=16,
      min_parse_tree_size=24,
      max_parse_tree_size=24,
      seed=2,
      model_name='gemini:gemini-3.1-flash-lite-preview',
      reasoning_effort='low',
    )
  )
  def test_sample_responses(
    self,
    n_query: int,
    min_parse_tree_size: int,
    max_parse_tree_size: int,
    seed: int,
    model_name: str,
    reasoning_effort: Literal['low'] | Literal['high'],
  ) -> None:
    model = APIModel(
      model_name,
      reasoning_effort=reasoning_effort,
      temperature=0,
    )

    dataloader, system_prompt, _ = boolean.create_boolean_dataloader(
      n_features=3,
      feature_maxval=3,
      n_shuffle_per_rule=1,
      n_query=n_query,
      min_parse_tree_size=min_parse_tree_size,
      max_parse_tree_size=max_parse_tree_size,
      n_rules_per_size=2,
      n_translations_per_rule=2,
      seed=seed,
    )

    print(system_prompt)
    for batch in dataloader:
      print(batch.prompt)
      response = model(batch.prompt, system_prompt)
      print(response.text())
      print(batch.target)
      print(batch.rule[0])
      break


if __name__ == '__main__':
  absltest.main()
