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

from collections import defaultdict

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from synthsys.data import listint
from synthsys.model.api import APIModel


class ListintTestCase(parameterized.TestCase):
  @parameterized.parameters(
    dict(n_support=8, n_query=8),
    dict(n_support=8, n_query=1),
    dict(n_support=16, n_query=16),
  )
  def test_list_function_generator_sample(self, n_support: int, n_query: int) -> None:
    """Test basic functionality of ListFunctionGenerator.sample()."""
    generator = listint.ListFunctionGenerator(
      list_maxlength=10, feature_maxval=10, seed=4
    )
    task = generator.sample(n_support=n_support, n_query=n_query)
    print(task.parse_tree)
    print(task.parse_tree.expression)
    print(task.parse_tree.probability)

    print(listint.format_as_prompt(task))

    # Check that we get the expected number of samples
    self.assertEqual(len(task.inputs_support), n_support)
    self.assertEqual(len(task.outputs_support), n_support)
    self.assertEqual(len(task.inputs_query), n_query)
    self.assertEqual(len(task.outputs_query), n_query)

  @parameterized.parameters(dict(n_translation_per_rule=3, n_shuffle_per_rule=4))
  def test_listint_dataloader(
    self, n_translation_per_rule: int, n_shuffle_per_rule: int
  ) -> None:
    dataloader, prompt, schema = listint.create_listint_dataloader(
      n_support=8,
      n_query=4,
      min_parse_tree_size=1,
      max_parse_tree_size=8,
      n_rules_per_size=10,
      n_translation_per_rule=n_translation_per_rule,
      n_shuffle_per_rule=n_shuffle_per_rule,
      list_maxlength=10,
      feature_maxval=10,
      easy_instructions=False,
      seed=0,
    )

    batches = list(dataloader)
    self.assertGreater(len(batches), 0)

    batches_by_set = defaultdict(list)
    for batch in batches:
      batches_by_set[batch.info['set_id'][0]].append(batch)

    for _, set_batches in batches_by_set.items():
      self.assertEqual(len(set_batches), n_shuffle_per_rule + n_translation_per_rule + 1)
      n_canonicals = sum(
        1 for b in set_batches if b.info['variation_type'][0] == 'canonical'
      )
      n_translations = sum(
        1 for b in set_batches if b.info['variation_type'][0] == 'translation'
      )
      n_shuffles = sum(1 for b in set_batches if b.info['variation_type'][0] == 'shuffle')
      self.assertEqual(n_canonicals, 1)
      self.assertEqual(n_translations, n_translation_per_rule)
      self.assertEqual(n_shuffles, n_shuffle_per_rule)

      probs = set(b.info['parse_tree_prob'][0] for b in set_batches)
      self.assertEqual(len(probs), 1)

  def test_listint_translator(self, max_parse_tree_size: int = 10, seed: int = 0) -> None:
    generator = listint.ListFunctionGenerator(
      list_maxlength=10, feature_maxval=10, seed=seed
    )
    translator = listint.Translator(seed=seed)

    for size in range(1, max_parse_tree_size):
      for parse_tree in generator.grammar.sample_by_size(size):
        task = generator.sample(n_support=8, n_query=5, parse_tree=parse_tree)
        translated_task = translator(task, generator, generator.feature_maxval)

        for input_list, output_list in zip(
          translated_task.inputs_support,
          translated_task.outputs_support,
          strict=True,
        ):
          translated_out = generator.evaluate(
            input_list, translated_task.parse_tree.expression
          )
          np.testing.assert_array_equal(translated_out, output_list)

        break  # NOTE: Comment to do test exhaustively

  @absltest.skip('Skipping long-running inspection test')
  @parameterized.parameters(
    dict(parse_tree_size=10),
    dict(parse_tree_size=11),
    dict(parse_tree_size=12),
  )
  def test_inspect_expressions(self, parse_tree_size: int) -> None:
    gen = listint.ListFunctionGenerator(list_maxlength=16, feature_maxval=10, seed=0)
    for tree in gen.grammar.sample_by_size(parse_tree_size):
      print(tree.expression)
      break

  @absltest.skip('Skipping long-running sample response test')
  @parameterized.parameters(
    dict(
      n_support=8,
      n_query=8,
      seed=0,
      model_name='gemini:gemini-3.1-flash-lite-preview',
      reasoning_effort='low',
    )
  )
  def test_sample_responses(
    self,
    n_support: int,
    n_query: int,
    seed: int,
    model_name: str,
    reasoning_effort: str,
  ) -> None:
    model = APIModel(
      model_name=model_name,
      reasoning_effort=reasoning_effort,
      temperature=0,
    )

    dataloader, system_prompt, _ = listint.create_listint_dataloader(
      n_support=n_support,
      n_query=n_query,
      min_parse_tree_size=10,
      max_parse_tree_size=12,
      n_rules_per_size=1,
      n_translation_per_rule=0,
      n_shuffle_per_rule=0,
      list_maxlength=16,
      feature_maxval=10,
      seed=seed,
    )
    print(system_prompt)
    for batch in dataloader:
      print(batch.rule[0])
      response = model(batch.prompt, system_prompt)
      print('\n===PROMPT===')
      print(batch.prompt)
      print('\n===RESPONSE===')
      print(response.text())
      print('\n===TARGET===')
      print(batch.target)
      break


if __name__ == '__main__':
  absltest.main()
