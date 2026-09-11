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

import itertools

import pandas as pd
import plotly.express as px
from absl.testing import absltest
from absl.testing import parameterized

from synthsys.data import mlc
from synthsys.model.api import APIModel


class MLCTestCase(parameterized.TestCase):
  @parameterized.parameters(
    dict(n_primitives=4, n_rules=2, n_support=2),
    dict(n_primitives=3, n_rules=3, n_support=2),
    dict(n_primitives=3, n_rules=4, n_support=2),
    dict(n_primitives=3, n_rules=5, n_support=2),
  )
  def test_meta_grammar_sample(
    self,
    n_primitives: int,
    n_rules: int,
    n_support: int,
  ) -> None:
    input_symbols, output_symbols = mlc.Vocabularies.LETTERS.value
    meta_grammar = mlc.MetaGrammar(
      n_primitives=n_primitives,
      n_rules=n_rules,
      input_vocab=input_symbols,
      output_vocab=output_symbols,
      max_rhs_length=6,
      max_example_length=8,
      p_lhs_onearg=0.5,
      seed=0,
    )
    language = meta_grammar.sample()

    self.assertEqual(len(language.primitives), n_primitives)
    self.assertEqual(len(language.rules), n_rules)

    episode = language.generate_episode(
      n_support=n_support, n_query=4, max_depth=16, max_tries=1000
    )

    self.assertGreaterEqual(len(episode.support_in), n_support)
    self.assertEqual(len(episode.support_out), len(episode.support_in))
    self.assertEqual(len(episode.query_in), 4)
    self.assertEqual(len(episode.query_out), 4)

  @parameterized.parameters(
    dict(n_support=0),
    dict(n_support=2),
    dict(n_support=4),
    dict(n_support=8),
    dict(n_support=16),
  )
  def test_create_mlc_dataloader(
    self,
    n_languages: int = 100,
    n_translation_per_language: int = 2,
    n_recomposition_per_language: int = 2,
    n_shuffle_per_language: int = 2,
    n_support: int = 0,
    n_query: int = 10,
  ) -> None:
    dataloader, prompt, _schema = mlc.create_mlc_dataloader(
      n_primitives=4,
      n_rules=3,
      vocabulary='MLC',
      max_rhs_length=3,
      max_example_length=20,
      p_lhs_onearg=0.5,
      n_languages=n_languages,
      n_translation_per_language=n_translation_per_language,
      n_recomposition_per_language=n_recomposition_per_language,
      n_shuffle_per_language=n_shuffle_per_language,
      n_support=n_support,
      n_query=n_query,
      seed=0,
    )
    self.assertIsInstance(dataloader, mlc.MLCDataloader)
    self.assertIn(' examples', prompt)
    self.assertIn(f'{n_query} new input', prompt)

    batches = list(dataloader)
    n_variations_per_language = (
      n_translation_per_language + n_recomposition_per_language + n_shuffle_per_language
    )
    self.assertLen(batches, n_languages * (1 + n_variations_per_language))

    for batch in batches:
      self.assertEqual(len(batch.target), n_query)
      self.assertEqual(len(batch.rule), n_query)
      self.assertEqual(len(batch.info['set_id']), n_query)
      self.assertEqual(len(batch.info['variation_type']), n_query)
      self.assertEqual(len(batch.info['variation_idx']), n_query)

  def test_parser_equivalence(self) -> None:
    input_symbols, output_symbols = mlc.Vocabularies.LETTERS.value
    meta_grammar = mlc.MetaGrammar(
      n_primitives=3,
      n_rules=5,
      input_vocab=input_symbols,
      output_vocab=output_symbols,
      max_rhs_length=4,
      max_example_length=50,
      p_lhs_onearg=0.5,
      seed=0,
    )
    language = meta_grammar.sample()

    for _ in range(1000):
      tree = language.sample_unambiguous_tree(max_depth=16, max_tries=10000)
      in_str = tree.render_input()
      self.assertEqual(
        tree.render_output(language.parser.primitives_map),
        language.parser.parse(in_str),
        f"Mismatch for input '{in_str}'.",
      )

  def test_shuffler(self) -> None:
    shuffler = mlc.Shuffler(seed=0)
    episode = mlc.LearningEpisode(
      support_in=['in1', 'in2', 'in3', 'in4', 'in5'],
      support_out=['out1', 'out2', 'out3', 'out4', 'out5'],
      query_in=['qin1', 'qin2', 'qin3', 'qin4', 'qin5'],
      query_out=['qout1', 'qout2', 'qout3', 'qout4', 'qout5'],
    )

    shuffled_episode = shuffler(episode)

    # Check that queries are correctly permuted as pairs and not lost
    orig_query_pairs = list(zip(episode.query_in, episode.query_out, strict=True))
    new_query_pairs = list(
      zip(shuffled_episode.query_in, shuffled_episode.query_out, strict=True)
    )
    self.assertCountEqual(new_query_pairs, orig_query_pairs)
    self.assertNotEqual(new_query_pairs, orig_query_pairs)

    # Check that support items are correctly permuted as pairs and not lost
    orig_support_pairs = list(zip(episode.support_in, episode.support_out, strict=True))
    new_support_pairs = list(
      zip(shuffled_episode.support_in, shuffled_episode.support_out, strict=True)
    )
    self.assertCountEqual(new_support_pairs, orig_support_pairs)
    self.assertNotEqual(new_support_pairs, orig_support_pairs)

  def test_recomposer(self) -> None:
    primitives = ['a', 'b', 'c', 'd']
    recomposer = mlc.Recomposer(seed=0)

    in_str = 'a func1 b func2 d func3 d'
    in_str_mod = recomposer(in_str, primitives)

    tokens_in = in_str.split()
    tokens_out = in_str_mod.split()

    self.assertEqual(len(tokens_in), len(tokens_out))

    # Check that non-primitives are unchanged
    for i, token in enumerate(tokens_in):
      if token not in primitives:
        self.assertEqual(tokens_out[i], token)

    # Check that primitives are a permutation
    prims_in = [t for t in tokens_in if t in primitives]
    prims_out = [t for t in tokens_out if t in primitives]
    self.assertCountEqual(prims_in, prims_out)

    # Check that it actually permuted
    self.assertNotEqual(in_str, in_str_mod)

  @absltest.skip('Skipping long-running visualization test')
  def test_task_difficulty(self, n_primitives: int = 3, n_languages: int = 5) -> None:
    n_rules_list = [3, 4, 5]
    max_rhs_length_list = [3, 4, 5]
    max_example_length_list = [16, 32]
    p_lhs_onearg_list = [0.1, 0.5, 0.9]

    input_symbols, output_symbols = mlc.Vocabularies.LETTERS.value

    results = []
    for n_rules, max_rhs, max_example, p_lhs in itertools.product(
      n_rules_list, max_rhs_length_list, max_example_length_list, p_lhs_onearg_list
    ):
      for i in range(n_languages):
        print(
          f'Running seed {i}, n_rules {n_rules}, max_rhs {max_rhs},'
          f'max_example {max_example}, p_lhs {p_lhs}'
        )
        try:
          meta_grammar = mlc.MetaGrammar(
            n_primitives=n_primitives,
            n_rules=n_rules,
            input_vocab=input_symbols,
            output_vocab=output_symbols,
            max_rhs_length=max_rhs,
            max_example_length=max_example,
            p_lhs_onearg=p_lhs,
            seed=i,
          )
          language = meta_grammar.sample()
          episode = language.generate_episode(n_support=2, n_query=10, max_tries=1000)

          for q_in, q_out in zip(episode.query_in, episode.query_out, strict=True):
            results.append(
              {
                'n_rules': n_rules,
                'max_rhs_length': max_rhs,
                'max_example_length': max_example,
                'p_lhs_onearg': p_lhs,
                'seed': i,
                'query_in_length': len(q_in.split()),
                'query_out_length': len(q_out.split()),
              }
            )
        except (RuntimeError, ValueError):
          print(
            f'Skipped seed {i}, n_rules {n_rules}, max_rhs {max_rhs},'
            f'max_example {max_example}, p_lhs {p_lhs}'
          )

    df = pd.DataFrame(results)
    fig = px.violin(
      df,
      x='max_example_length',
      y='query_out_length',
      color='p_lhs_onearg',
      facet_col='n_rules',
      facet_row='max_rhs_length',
      title=f'n_primitives={n_primitives}',
      box=True,
    )
    fig.show()
    # fig.write_html('mlc_difficulty.html')

  @absltest.skip('Skipping long-running sample response test')
  @parameterized.parameters(
    dict(
      n_support=2,
      n_query=10,
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

    dataloader, system_prompt, _ = mlc.create_mlc_dataloader(
      n_primitives=4,
      n_rules=3,
      vocabulary='MLC',
      max_rhs_length=3,
      max_example_length=16,
      p_lhs_onearg=0.5,
      n_languages=1,
      n_translation_per_language=0,
      n_recomposition_per_language=0,
      n_shuffle_per_language=0,
      n_support=n_support,
      n_query=n_query,
      seed=seed,
    )
    print(system_prompt)
    for batch in dataloader:
      response = model(batch.prompt, system_prompt)
      print('\n===PROMPT===')
      print(batch.prompt)
      print('\n===RESPONSE===')
      print(response.text())
      print('\n===TARGET===')
      print(batch.target)


if __name__ == '__main__':
  absltest.main()
