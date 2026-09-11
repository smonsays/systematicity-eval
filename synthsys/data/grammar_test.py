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
from collections import namedtuple

from absl.testing import absltest
from absl.testing import parameterized

from synthsys.data.grammar import LiteralNode
from synthsys.data.grammar import Nonterminal
from synthsys.data.grammar import ProbabilisticContextFreeGrammar
from synthsys.data.grammar import Production
from synthsys.data.grammar import VariableNode


class GrammarTestCase(parameterized.TestCase):
  def test_simple_grammar_generation(self) -> None:
    """Test basic functionality of ProbabilisticContextFree"""
    # Create simple grammar as in the original main example
    A = Nonterminal('A')
    productions = [
      Production(A, (A, VariableNode('b', int)), 0.6, 'add'),
      Production(A, (VariableNode('a', int),), 0.2),
      Production(A, (LiteralNode('1', int),), 0.2),
    ]

    seed = 3
    pcfg = ProbabilisticContextFreeGrammar(productions, A, seed=seed)
    parse_tree = pcfg.sample()
    print('Simple example:')
    print(parse_tree.expression)
    print(parse_tree.probability)
    print(parse_tree)

    # Check for determinism and hashability
    pcfg = ProbabilisticContextFreeGrammar(productions, A, seed=seed)
    assert parse_tree == pcfg.sample()

  def test_grammar_samples_in_enumerate(self, n_samples: int = 1000) -> None:
    """Test whether samples from pcfg.sample() are contained in pcfg.enumerate().

    This is using a test version of the Boolean grammar to make the test case
    sufficiently interesting.
    """
    GeometricObject = namedtuple('GeometricObject', ['shape', 'color', 'size'])
    START = Nonterminal('START')
    BOOL = Nonterminal('BOOL')
    PREDICATE = Nonterminal('PREDICATE')
    OBJECT = Nonterminal('OBJECT')
    COLOR = Nonterminal('COLOR', allow_resample=True)
    SHAPE = Nonterminal('SHAPE', allow_resample=True)
    SIZE = Nonterminal('SIZE', allow_resample=True)

    prob_basic = 1.0 / 12.0
    prob_bool = 1.0 / 13.0

    shapes = ['square', 'circle', 'triangle']
    colors = ['red', 'blue', 'yellow']
    sizes = ['tiny', 'small', 'large']

    productions = [
      Production(START, (OBJECT,), prob_basic, fun_name='true_'),
      Production(START, (OBJECT,), prob_basic, fun_name='false_'),
      Production(START, (BOOL,), 1.0 - 2 * prob_basic),
      Production(BOOL, (BOOL, BOOL), prob_bool, fun_name='and_'),
      Production(BOOL, (BOOL, BOOL), prob_bool, fun_name='or_'),
      Production(BOOL, (BOOL,), prob_bool, fun_name='not_'),
      Production(BOOL, (PREDICATE,), 1.0 - 3 * prob_bool),
      Production(OBJECT, (VariableNode('x', GeometricObject),), 1.0),
      Production(PREDICATE, (OBJECT, COLOR), 1.0 / 3.0, fun_name='is_color_'),
      Production(PREDICATE, (OBJECT, SHAPE), 1.0 / 3.0, fun_name='is_shape_'),
      Production(PREDICATE, (OBJECT, SIZE), 1.0 / 3.0, fun_name='is_size_'),
      *[Production(COLOR, (LiteralNode(f"'{c}'", str),), 1.0 / 3.0) for c in colors],
      *[Production(SHAPE, (LiteralNode(f"'{s}'", str),), 1.0 / 3.0) for s in shapes],
      *[Production(SIZE, (LiteralNode(f"'{s}'", str),), 1.0 / 3.0) for s in sizes],
    ]

    pcfg = ProbabilisticContextFreeGrammar(productions, START, 0)
    enumerated_expressions = set()
    max_size = 0
    for parse_tree in itertools.islice(pcfg.enumerate(), n_samples):
      enumerated_expressions.add(parse_tree.expression)
      max_size = len(parse_tree)

    for _ in range(n_samples):
      parse_tree = pcfg.sample()
      if len(parse_tree) < max_size:
        self.assertIn(
          parse_tree.expression,
          enumerated_expressions,
          f"Sample with expression '{parse_tree.expression}' not found in enumerated set",
        )

  def test_grammar_enumerate(self) -> None:
    A = Nonterminal('A')
    B = Nonterminal('B')
    productions = [
      Production(A, (A, B), 0.6, 'add'),
      Production(A, (LiteralNode('1', int),), 0.4),
      Production(B, (LiteralNode('2', int),), 0.5),
      Production(B, (LiteralNode('3', int),), 0.5),
    ]
    seed = 3
    pcfg = ProbabilisticContextFreeGrammar(productions, A, seed=seed)

    last_len = 0
    for parse_tree in itertools.islice(pcfg.enumerate(), 3):
      assert len(parse_tree) >= last_len
      last_len = len(parse_tree)
      if 'add' in parse_tree.expression:
        assert 'B' in repr(parse_tree)
      print(parse_tree.expression)
      print(parse_tree)

  def test_resample_respects_allow_resample(self) -> None:
    """Test that `resample` only modifies Nonterminals with allow_resample=True."""
    # Create grammar: A (non-resampleable) -> add(B (resampleable), 'a')
    A = Nonterminal('A', allow_resample=False)
    B = Nonterminal('B', allow_resample=True)

    productions = [
      Production(A, (B, VariableNode('a', int)), 1.0, 'add'),
      Production(B, (A,), 0.1),
      Production(B, (LiteralNode('1', int),), 0.45),
      Production(B, (LiteralNode('2', int),), 0.45),
    ]

    pcfg = ProbabilisticContextFreeGrammar(productions, A, seed=0)
    original_tree = pcfg.sample()

    # Test that non-resampleable A preserves its structure
    resampled = pcfg.resample(original_tree)
    self.assertEqual(original_tree.symbol, resampled.symbol)
    self.assertEqual(original_tree.production_used, resampled.production_used)
    self.assertIsNotNone(resampled.children)
    self.assertEqual(len(resampled.children), 2)

    # Test that resampleable B can change values across multiple regenerations
    b_values_seen = set()
    for _ in range(10):
      resampled = pcfg.resample(original_tree)
      b_child = resampled.children[0]
      b_values_seen.add(b_child.expression)

    # With allow_resample=True, B should produce different values
    self.assertGreater(len(b_values_seen), 1)

  def test_is_finite(self) -> None:
    """Test that is_finite correctly detects finite vs infinite grammars."""
    # Finite grammar: no cycles
    A = Nonterminal('A')
    finite_productions = [
      Production(A, (LiteralNode('1', int),), 0.5),
      Production(A, (LiteralNode('2', int),), 0.5),
    ]
    with self.assertRaises(ValueError):
      ProbabilisticContextFreeGrammar(finite_productions, A, seed=0)

    # Infinite grammar: A -> A (direct cycle)
    infinite_productions = [
      Production(A, (A,), 0.5),
      Production(A, (LiteralNode('1', int),), 0.5),
    ]
    pcfg = ProbabilisticContextFreeGrammar(infinite_productions, A, seed=0)
    self.assertFalse(pcfg.is_finite())

  def test_sample_by_size(self, max_parse_tree_size: int = 100) -> None:
    """Test that sample_by_size generates parse trees of the correct size."""
    A = Nonterminal('A')
    productions = [
      Production(A, (A, VariableNode('b', int)), 0.6, 'add'),
      Production(A, (VariableNode('a', int),), 0.2),
      Production(A, (LiteralNode('1', int),), 0.2),
    ]
    pcfg = ProbabilisticContextFreeGrammar(
      productions, A, seed=0, max_parse_tree_size=max_parse_tree_size
    )

    # Test that all sampled trees have the requested size
    for target_size in range(2, max_parse_tree_size):
      print(target_size)
      for parse_tree in pcfg.sample_by_size(target_size):
        self.assertEqual(len(parse_tree), target_size)
        self.assertTrue(parse_tree.is_complete())

  def test_dp_counts_match_generated_trees(self) -> None:
    """Test that the DP counts precisely match the number of valid parse trees."""
    A = Nonterminal('A')
    B = Nonterminal('B')
    productions = [
      Production(A, (A, B), 0.4, 'add'),
      Production(A, (LiteralNode('1', int),), 0.3),
      Production(A, (VariableNode('x', int),), 0.3),
      Production(B, (B, B), 0.2, 'mul'),
      Production(B, (A,), 0.4, 'ident'),
      Production(B, (LiteralNode('2', int),), 0.4),
    ]
    max_size = 8
    pcfg = ProbabilisticContextFreeGrammar(
      productions, A, seed=0, max_parse_tree_size=max_size
    )

    for target_size in range(1, max_size + 1):
      expected_count = pcfg.counts[A].get(target_size, 0)

      generated_trees = set()
      for index in range(expected_count):
        tree = pcfg._build_tree_at_index(A, target_size, index, pcfg.counts)
        self.assertEqual(len(tree), target_size)
        self.assertTrue(tree.is_complete())
        generated_trees.add(tree)

      self.assertEqual(len(generated_trees), expected_count)

      # Verify that requesting out of bounds raises IndexError
      with self.assertRaises(IndexError):
        pcfg._build_tree_at_index(A, target_size, expected_count, pcfg.counts)


if __name__ == '__main__':
  absltest.main()
