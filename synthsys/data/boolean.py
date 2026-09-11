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

"""Data generation for boolean rule induction tasks."""

import dataclasses
import hashlib
from typing import Iterator

import einops
import jaxtyping as jt
import numpy as np
import pydantic

from synthsys.data.base import Batch
from synthsys.data.grammar import LiteralNode
from synthsys.data.grammar import Nonterminal
from synthsys.data.grammar import ParseTree
from synthsys.data.grammar import ProbabilisticContextFreeGrammar
from synthsys.data.grammar import Production
from synthsys.data.grammar import VariableNode

FEATURE_POOLS = {
  'numerosity': ('one', 'two', 'three', 'four', 'five'),
  'size': ('tiny', 'small', 'medium', 'large', 'huge'),
  'color': ('blue', 'yellow', 'green', 'red', 'black'),
  'material': ('glass', 'metal', 'wood', 'plastic', 'ceramic'),
  'shape': ('circle', 'triangle', 'square', 'pentagon', 'star'),
}

SYSTEM_PROMPT_TEMPLATE = """
Your task is to discover the meaning of 'wudsy', a word in an alien language.
In the following you will be presented with several examples of objects that are either wudsy or not wudsy.
Afterwards you will be presented with {n_query} new object(s) and you have to determine whether they are either wudsy or not wudsy.
Try to identify the simplest, deterministic rule for what makes objects wudsy or not wudsy and apply it to decide for the new objects.

Each rule is generated from a context-free grammar. The specific grammar and properties of the objects will be provided for each task. The shortest generated rule that explains the examples is the correct one to base the decision on.

Note in particular that this grammar does not contain a not() operation, keep this in mind when trying to identify the "simplest" rule, i.e. the shortest rule that explains all examples that can be expressed in this grammar.

Respond with the boolean value for each of the {n_query} new object(s) in their presented order according to the rule you identified.
"""  # noqa: E501


@dataclasses.dataclass(frozen=True)
class Vocabulary:
  feature_names: list[str]
  feature_values: list[tuple[str, ...]]

  @classmethod
  def sample(cls, seed: int, n_features: int, feature_maxval: int) -> 'Vocabulary':
    rng = np.random.default_rng(seed)
    if n_features > len(FEATURE_POOLS):
      raise ValueError(f'n_features must be <= {len(FEATURE_POOLS)}')
    max_feature_val = min(len(pool) for pool in FEATURE_POOLS.values())
    if feature_maxval > max_feature_val:
      raise ValueError(f'feature_maxval must be <= {max_feature_val}')

    other_features = [f for f in FEATURE_POOLS.keys() if f != 'shape']
    selected_features = [
      'shape',
      *rng.choice(other_features, size=n_features - 1, replace=False).tolist(),
    ]
    feature_values = []
    for feat in selected_features:
      pool = FEATURE_POOLS[feat]
      sampled_vals = tuple(rng.choice(pool, size=feature_maxval, replace=False).tolist())
      feature_values.append(sampled_vals)

    return cls(feature_names=selected_features, feature_values=feature_values)

  def render_object(self, obj_array: jt.Int[np.ndarray, ' feature']) -> tuple[str, bool]:
    words = []
    has_numerosity = 'numerosity' in self.feature_names
    is_plural = False

    for feat_name in FEATURE_POOLS.keys():
      if feat_name in self.feature_names:
        idx = self.feature_names.index(feat_name)
        val_word = self.feature_values[idx][obj_array[idx]]
        words.append(val_word)
        if feat_name == 'numerosity' and val_word != 'one':
          is_plural = True

    if 'shape' not in self.feature_names:
      words.append('object')

    if is_plural:
      words[-1] += 's'

    prefix = '' if has_numerosity else ('an ' if words[0][0] in 'aeiou' else 'a ')
    return f'{prefix}{" ".join(words)}', is_plural

  def get_task_description(self) -> str:
    lines = []

    descriptions = []
    for feat_name in FEATURE_POOLS.keys():
      if feat_name in self.feature_names:
        idx = self.feature_names.index(feat_name)
        vals = self.feature_values[idx]
        vals_str = (
          ', '.join(vals[:-1]) + (' or ' if len(vals) > 1 else '') + vals[-1]
          if len(vals) > 1
          else vals[0]
        )
        descriptions.append(f'a {feat_name} ({vals_str})')

    desc_str = ''
    if len(descriptions) > 1:
      desc_str = ', '.join(descriptions[:-1]) + ' and ' + descriptions[-1]
    elif len(descriptions) == 1:
      desc_str = descriptions[0]

    lines.append(f'Each object has {desc_str}.')
    lines.append('The rules are generated from the following context-free grammar:\n')
    lines.append('```')

    is_funcs = [
      f'is_{feat}(O, {chr(65 + i)})' for i, feat in enumerate(self.feature_names)
    ]
    funcs_str = ' | '.join(is_funcs)
    lines.append(f"S -> 'true' | 'false' | and(S, S) | or(S, S) | {funcs_str}")
    lines.append("O -> 'x'")

    for i, _feat in enumerate(self.feature_names):
      vals = self.feature_values[i]
      vals_str = ' | '.join(f"'{v}'" for v in vals)
      lines.append(f'{chr(65 + i)} -> {vals_str}')

    lines.append('```')
    return '\n'.join(lines)


@dataclasses.dataclass(frozen=True)
class BooleanTask:
  inputs_support: jt.Int[np.ndarray, 'n_support feature']
  outputs_support: jt.Bool[np.ndarray, ' n_support']
  inputs_query: jt.Int[np.ndarray, 'n_query feature']
  outputs_query: jt.Bool[np.ndarray, ' n_query']
  output_table: jt.Bool[np.ndarray, ' n_all']
  parse_tree: ParseTree


class BooleanConceptGenerator:
  """Boolean concept learning tasks."""

  def __init__(self, n_features: int, feature_maxval: int, seed: int) -> None:
    self.rng = np.random.default_rng(seed)
    self.n_features = n_features
    self.feature_maxval = feature_maxval

    START = Nonterminal('START')
    BOOL = Nonterminal('BOOL')
    OBJECT = Nonterminal('OBJECT')
    FEATURE_IDX = Nonterminal('FEATURE_IDX')
    FEATURE_VAL = Nonterminal('FEATURE_VAL')

    # Production probabilities are computed to ensure that the probability of an
    # expression is a strictly monotonically decreasing function of its length
    productions = [
      Production(START, (OBJECT,), 1.0 / 2.25, fun_name='true_'),
      Production(START, (OBJECT,), 1.0 / 2.25, fun_name='false_'),
      Production(START, (BOOL,), 0.25 / 2.25),
      Production(BOOL, (BOOL, BOOL), 0.0625 / 3.125, fun_name='and_'),
      Production(BOOL, (BOOL, BOOL), 0.0625 / 3.125, fun_name='or_'),
      Production(
        BOOL, (OBJECT, FEATURE_IDX, FEATURE_VAL), 3.0 / 3.125, fun_name='is_feature_'
      ),
      Production(OBJECT, (VariableNode('x', np.ndarray),), 1.0),
      *[
        Production(FEATURE_IDX, (LiteralNode(str(i), int),), 1.0 / n_features)
        for i in range(n_features)
      ],
      *[
        Production(FEATURE_VAL, (LiteralNode(str(i), int),), 1.0 / feature_maxval)
        for i in range(feature_maxval)
      ],
    ]
    self.grammar = ProbabilisticContextFreeGrammar(productions, START, seed)

    # All possible objects (cartesian product)
    shape_args = [feature_maxval] * n_features
    self.all_objects = einops.rearrange(np.indices(shape_args), 'f ... -> (...) f')

  def sample(
    self, n_support: int, n_query: int, parse_tree: ParseTree | None = None
  ) -> BooleanTask:
    assert (n_support + n_query) <= len(self.all_objects)

    if parse_tree is None:
      parse_tree = self.grammar.sample()

    exp = parse_tree.expression
    output_table = self.evaluate(self.all_objects, exp)

    support_indices = self.rng.choice(
      len(self.all_objects), size=n_support, replace=False
    )
    remaining_indices = np.setdiff1d(
      np.arange(len(self.all_objects)), support_indices, assume_unique=True
    )
    query_indices = self.rng.choice(remaining_indices, size=n_query, replace=False)

    return BooleanTask(
      inputs_support=self.all_objects[support_indices],
      outputs_support=output_table[support_indices],
      inputs_query=self.all_objects[query_indices],
      outputs_query=output_table[query_indices],
      output_table=output_table,
      parse_tree=parse_tree,
    )

  def sample_curriculum(
    self, n_query: int, parse_tree: ParseTree | None = None
  ) -> BooleanTask:
    """Create a curriculum of support examples that attempt to uniquely idenitfy the rule.

    This is a heuristic and it is not guaranteed that there isn't another, simpler rule
    that would equally explain the support examples. The heuristic works by creating the
    Hamming graph over objects, i.e. there is an edge if two objects differ by a single
    attribute and adding all objects that sit on the decision boundary to the support set.
    """

    if parse_tree is None:
      parse_tree = self.grammar.sample()

    exp = parse_tree.expression
    output_table = self.evaluate(self.all_objects, exp)
    n_all_objects = len(self.all_objects)

    # Construct the hamming graph over objects coloured by their label
    diffs = (self.all_objects[:, None, :] != self.all_objects[None, :, :]).sum(axis=-1)
    adjacency_matrix = diffs == 1

    # Add all nodes to the suppport set that sit on the decision boundary
    support_set = set()
    for u in range(n_all_objects):
      for v in np.where(adjacency_matrix[u])[0]:
        if output_table[u] != output_table[v]:
          support_set.add(u)
          support_set.add(v)

    if len(support_set) == 0:
      # Edge case: All True or all False
      anchor = self.rng.choice(n_all_objects)
      support_set.add(anchor)
      neighbors = np.where(adjacency_matrix[anchor])[0]
      support_set.update(neighbors.tolist())

    # Shuffle support and query indices
    support_indices = np.array(list(support_set), dtype=int)
    support_indices = self.rng.permuted(support_indices)
    remaining_indices = np.setdiff1d(np.arange(n_all_objects), support_indices)

    if len(remaining_indices) < n_query:
      raise ValueError(
        f'Not enough remaining objects for query set.'
        f'Needed {n_query}, got {len(remaining_indices)}'
      )

    query_indices = self.rng.choice(remaining_indices, size=n_query, replace=False)

    return BooleanTask(
      inputs_support=self.all_objects[support_indices],
      outputs_support=output_table[support_indices],
      inputs_query=self.all_objects[query_indices],
      outputs_query=output_table[query_indices],
      output_table=output_table,
      parse_tree=parse_tree,
    )

  def evaluate(
    self, input: jt.Int[np.ndarray, '... feature'], expression: str
  ) -> jt.Bool[np.ndarray, '...']:
    def and_(
      a: jt.Bool[np.ndarray, '...'], b: jt.Bool[np.ndarray, '...']
    ) -> jt.Bool[np.ndarray, '...']:
      return np.logical_and(a, b)

    def or_(
      a: jt.Bool[np.ndarray, '...'], b: jt.Bool[np.ndarray, '...']
    ) -> jt.Bool[np.ndarray, '...']:
      return np.logical_or(a, b)

    def not_(a: jt.Bool[np.ndarray, '...']) -> jt.Bool[np.ndarray, '...']:
      return np.logical_not(a)

    def is_feature_(
      obj: jt.Int[np.ndarray, '... feature'], idx: int, val: int
    ) -> jt.Bool[np.ndarray, '...']:
      return obj[..., idx] == val

    def true_(obj: jt.Int[np.ndarray, '... feature']) -> jt.Bool[np.ndarray, '...']:
      return np.ones(obj.shape[:-1], dtype=bool)

    def false_(obj: jt.Int[np.ndarray, '... feature']) -> jt.Bool[np.ndarray, '...']:
      return np.zeros(obj.shape[:-1], dtype=bool)

    functions = {
      'and_': and_,
      'or_': or_,
      'not_': not_,
      'is_feature_': is_feature_,
      'true_': true_,
      'false_': false_,
    }

    return eval(expression, {'__builtins__': {}}, dict(x=input) | functions)


@dataclasses.dataclass
class MaximumAposterioriPrediction:
  targets: jt.Bool[np.ndarray, ' n_query']
  prior_prob: float
  parse_tree_size: int
  expression: str
  is_unique: bool


class BooleanIdealObserver:
  def __init__(
    self, n_features: int, feature_maxval: int, max_parse_tree_size: int, seed: int
  ) -> None:
    self.generator = BooleanConceptGenerator(n_features, feature_maxval, seed=seed)
    hypotheses: dict[str, tuple[np.ndarray, float, int, str]] = dict()

    for i, parse_tree in enumerate(self.generator.grammar.enumerate()):
      task = self.generator.sample(n_support=0, n_query=0, parse_tree=parse_tree)
      expr = task.parse_tree.expression
      hypotheses[expr] = (
        task.output_table,
        task.parse_tree.probability,
        len(task.parse_tree),
        expr,
      )
      if len(task.parse_tree) > max_parse_tree_size:
        break
      if i > 1000000:
        raise RuntimeError('Ideal observer exceeded maximum hypotheses (1000000)')
    self.output_tables, self.priors, self.parse_tree_sizes, self.expressions = (
      np.array(arr) for arr in zip(*hypotheses.values(), strict=True)
    )
    # Cast expression to string since short expressions might be stored as dtype='<U20'
    self.expressions = self.expressions.astype(str)

  def maximum_aposteriori(
    self,
    inputs_support: jt.Int[np.ndarray, 'n_support feature'],
    outputs_support: jt.Bool[np.ndarray, ' n_support'],
    inputs_query: jt.Int[np.ndarray, 'n_query feature'],
  ) -> MaximumAposterioriPrediction:
    def find_matching_object_indices(
      x: jt.Int[np.ndarray, 'sample feature'],
    ) -> jt.Int[np.ndarray, ' sample']:
      return np.where((self.generator.all_objects[None, :] == x[:, None]).all(axis=-1))[1]

    indices_support = find_matching_object_indices(inputs_support)
    preds_support = self.output_tables[:, indices_support]
    posteriors = np.all(preds_support == outputs_support, axis=-1) * self.priors

    indices_query = find_matching_object_indices(inputs_query)
    map_idx = int(np.argmax(posteriors))

    map_indices = np.where(np.isclose(posteriors, posteriors[map_idx]))[0]
    all_map_preds = self.output_tables[map_indices][:, indices_query]
    map_is_unique = bool(np.all(all_map_preds == all_map_preds[0], axis=0).all())

    preds_query = self.output_tables[map_idx, indices_query]

    return MaximumAposterioriPrediction(
      targets=preds_query,
      prior_prob=float(self.priors[map_idx]),
      parse_tree_size=int(self.parse_tree_sizes[map_idx]),
      expression=str(self.expressions[map_idx]),
      is_unique=map_is_unique,
    )


def format_as_prompt(
  inputs_support: jt.Int[np.ndarray, 'n_support feature'],
  outputs_support: jt.Bool[np.ndarray, ' n_support'],
  inputs_query: jt.Int[np.ndarray, 'n_query feature'],
  vocab: Vocabulary,
) -> str:
  assert len(inputs_support) == len(outputs_support)
  prompt_lines = []

  prompt_lines.append(vocab.get_task_description())
  prompt_lines.append('')

  for i, (input, output) in enumerate(zip(inputs_support, outputs_support, strict=True)):
    obj_str, is_plural = vocab.render_object(input)
    label = 'wudsy' if output else 'not wudsy'
    verb = 'are' if is_plural else 'is'
    prompt_line = f'Example {i + 1}: {obj_str.capitalize()} {verb} {label}.'
    prompt_lines.append(prompt_line)

  prompt_lines.append('')

  for i, input in enumerate(inputs_query):
    obj_str, is_plural = vocab.render_object(input)
    verb_cap = 'Are' if is_plural else 'Is'
    prompt_line = f'Test {i + 1}: {verb_cap} {obj_str} wudsy?'
    prompt_lines.append(prompt_line)

  return '\n'.join(prompt_lines)


class Shuffler:
  def __init__(self, seed: int | None) -> None:
    self.rng = np.random.default_rng(seed)

  def __call__(self, task: BooleanTask) -> BooleanTask:
    def _shuffle(
      inputs: np.ndarray, outputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
      indices = self.rng.permutation(len(inputs))
      return inputs[indices], outputs[indices]

    inputs_support, outputs_support = _shuffle(task.inputs_support, task.outputs_support)
    inputs_query, outputs_query = _shuffle(task.inputs_query, task.outputs_query)

    return BooleanTask(
      inputs_support=inputs_support,
      outputs_support=outputs_support,
      inputs_query=inputs_query,
      outputs_query=outputs_query,
      output_table=task.output_table,
      parse_tree=task.parse_tree,
    )


class BooleanDataloader:
  def __init__(
    self,
    n_features: int,
    feature_maxval: int,
    n_query: int,
    min_parse_tree_size: int,
    max_parse_tree_size: int,
    n_rules_per_size: int,
    n_translations_per_rule: int,
    n_shuffle_per_rule: int,
    seed: int,
  ) -> None:
    self.n_features = n_features
    self.feature_maxval = feature_maxval
    self.n_rules_per_size = n_rules_per_size
    self.n_translations_per_rule = n_translations_per_rule
    self.n_shuffle_per_rule = n_shuffle_per_rule
    self.n_query = n_query
    self.min_parse_tree_size = min_parse_tree_size
    self.max_parse_tree_size = max_parse_tree_size
    self.rng = np.random.default_rng(seed)

    self.generator = BooleanConceptGenerator(n_features, feature_maxval, seed)
    self.ideal_observer = BooleanIdealObserver(
      n_features, feature_maxval, self.max_parse_tree_size + 1, seed + 1
    )
    self.shuffler = Shuffler(seed)

  def generate_tasks_by_size(
    self,
  ) -> Iterator[tuple[BooleanTask, MaximumAposterioriPrediction]]:
    seen_rules: set[bytes] = set()

    for size in range(self.min_parse_tree_size, self.max_parse_tree_size + 1):
      tasks_generated_for_current_size = 0
      for parse_tree in self.generator.grammar.sample_by_size(size):
        if tasks_generated_for_current_size >= self.n_rules_per_size:
          break

        try:
          task = self.generator.sample_curriculum(self.n_query, parse_tree)
        except ValueError:
          continue

        map_prediction = self.ideal_observer.maximum_aposteriori(
          task.inputs_support, task.outputs_support, task.inputs_query
        )

        # Ensure task is unambiguous
        if not np.all(task.outputs_query == map_prediction.targets):
          continue

        # Ensure task is functionally novel
        output_bytes = task.output_table.tobytes()
        if output_bytes in seen_rules:
          continue
        seen_rules.add(output_bytes)

        tasks_generated_for_current_size += 1
        yield task, map_prediction

  def package_batch(
    self,
    task: BooleanTask,
    map_prediction: MaximumAposterioriPrediction,
    vocab: Vocabulary,
    set_id: str,
    variation_type: str,
    variation_idx: int,
  ) -> Batch:
    output_table = ''.join(map(str, task.output_table.astype(int)))

    return Batch(
      prompt=format_as_prompt(
        task.inputs_support, task.outputs_support, task.inputs_query, vocab
      ),
      target=np.array(task.outputs_query),
      rule=np.array([task.parse_tree.expression] * self.n_query),
      info=dict(
        set_id=np.array([set_id] * self.n_query),
        variation_type=np.array([variation_type] * self.n_query),
        variation_idx=np.array([variation_idx] * self.n_query),
        output_table=np.array([output_table] * self.n_query),
        parse_tree_size=np.array([len(task.parse_tree)] * self.n_query),
        map_target=map_prediction.targets,
        map_parse_tree_size=np.array([map_prediction.parse_tree_size] * self.n_query),
        map_prior_prob=np.array([map_prediction.prior_prob] * self.n_query),
        map_rule=np.array([map_prediction.expression] * self.n_query),
        map_is_unique=np.array([map_prediction.is_unique] * self.n_query),
      ),
    )

  def __iter__(self) -> Iterator[Batch]:
    for task, map_prediction in self.generate_tasks_by_size():
      set_id = hashlib.md5(task.parse_tree.expression.encode('utf-8')).hexdigest()

      vocab_seed = int(self.rng.integers(0, 2**32 - 1))
      base_vocab = Vocabulary.sample(vocab_seed, self.n_features, self.feature_maxval)

      yield self.package_batch(
        task=task,
        map_prediction=map_prediction,
        vocab=base_vocab,
        set_id=set_id,
        variation_type='canonical',
        variation_idx=0,
      )

      # Create translation variations
      for i in range(self.n_translations_per_rule):
        vocab_seed = int(self.rng.integers(0, 2**32 - 1))
        vocab = Vocabulary.sample(vocab_seed, self.n_features, self.feature_maxval)

        yield self.package_batch(
          task=task,
          map_prediction=map_prediction,
          vocab=vocab,
          set_id=set_id,
          variation_type='translation',
          variation_idx=i + 1,
        )

      # Create shuffle variations
      for i in range(self.n_shuffle_per_rule):
        shuffled_task = self.shuffler(task)
        yield self.package_batch(
          task=shuffled_task,
          map_prediction=map_prediction,
          vocab=base_vocab,
          set_id=set_id,
          variation_type='shuffle',
          variation_idx=i + 1,
        )


def create_boolean_dataloader(
  n_features: int,
  feature_maxval: int,
  n_query: int,
  min_parse_tree_size: int,
  max_parse_tree_size: int,
  n_rules_per_size: int,
  n_translations_per_rule: int,
  n_shuffle_per_rule: int = 0,
  *,
  seed: int,
) -> tuple[BooleanDataloader, str, type[pydantic.BaseModel]]:
  dataloader = BooleanDataloader(
    n_features,
    feature_maxval,
    n_query,
    min_parse_tree_size,
    max_parse_tree_size,
    n_rules_per_size,
    n_translations_per_rule,
    n_shuffle_per_rule,
    seed,
  )
  system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n_query=n_query)
  schema = pydantic.create_model('Answer', **{str(i + 1): bool for i in range(n_query)})

  return (dataloader, system_prompt, schema)
