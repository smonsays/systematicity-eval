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

"""Data generation for listint rule induction tasks."""

import dataclasses
import hashlib
import math
from typing import Iterator

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

SYSTEM_PROMPT_TEMPLATE = """
Your task is to discover the rule that transforms input lists of integers into output lists of integers.
In the following, you will be presented with {n_support} examples of input and output list pairs.
Afterwards, you will be presented with {n_query} new input list(s) and you have to apply the inferred rule to predict their corresponding output lists.
{operations_description}
"""  # noqa: E501

OPERATIONS_DESCRIPTION = """

The rule is a composition of the following basic operations:
- insert: Insert an element at a specific position.
- remove: Remove an element at a specific position.
- repeat: Repeat the list until it reaches a specified length.
- shift: Shift all elements by a given offset, wrapping around.
- swap: Swap the first and second halves of the list.
- tail: Keep only a specified number of elements from the end of the list.
"""


@dataclasses.dataclass(frozen=True)
class ListFunctionTask:
  inputs_support: tuple[jt.Int[np.ndarray, ' input_length'], ...]
  outputs_support: tuple[jt.Int[np.ndarray, ' output_length'], ...]
  inputs_query: tuple[jt.Int[np.ndarray, ' input_length'], ...]
  outputs_query: tuple[jt.Int[np.ndarray, ' output_length'], ...]
  parse_tree: ParseTree


class ListFunctionGenerator:
  """List function learning tasks."""

  def __init__(self, seed: int, list_maxlength: int, feature_maxval: int) -> None:
    self.rng = np.random.default_rng(seed)
    self.list_maxlength = list_maxlength
    self.feature_maxval = feature_maxval

    START = Nonterminal('START')
    LIST = Nonterminal('LIST')

    POS = Nonterminal('POS', allow_resample=True)
    ELEM = Nonterminal('ELEM', allow_resample=True)
    LEN = Nonterminal('LEN', allow_resample=True)
    LEN_REPEAT = Nonterminal('LEN_REPEAT', allow_resample=True)

    # The probabilities should be chosen such that they sum to one and shorter expressions
    # have a strightly higher probability than longer ones
    p_endo = 1.0 / 6.0
    p_param = 1.0 / 3.0

    productions = [
      Production(START, (LIST,), 1.0),
      # Endomorphisms
      Production(LIST, (LIST,), p_endo, fun_name='swap_'),
      # Parameterized
      Production(LIST, (LIST, POS), p_param / 5.0, fun_name='remove_'),
      Production(LIST, (LIST, POS, ELEM), p_param / 5.0, fun_name='insert_'),
      Production(LIST, (LIST, LEN), p_param / 5.0, fun_name='tail_'),
      Production(LIST, (LIST, LEN_REPEAT), p_param / 5.0, fun_name='repeat_'),
      Production(LIST, (LIST, POS), p_param / 5.0, fun_name='shift_'),
      # Input
      Production(LIST, (VariableNode('x', np.ndarray),), 1.0 - p_endo - p_param),
      # Terminal Parameter Values
      *[Production(POS, (LiteralNode(str(p), int),), 1.0 / 7.0) for p in range(-3, 4)],
      *[
        Production(ELEM, (LiteralNode(str(e), int),), 1.0 / (self.feature_maxval + 1))
        for e in range(self.feature_maxval + 1)
      ],
      *[Production(LEN, (LiteralNode(str(l), int),), 1.0 / 3.0) for l in range(1, 4)],
      *[
        Production(LEN_REPEAT, (LiteralNode(str(l), int),), 1.0 / 5.0)
        for l in range(5, 10)
      ],
    ]
    self.grammar = ProbabilisticContextFreeGrammar(productions, START, seed)

  def sample(
    self, n_support: int, n_query: int, parse_tree: ParseTree | None = None
  ) -> ListFunctionTask:
    if parse_tree is None:
      parse_tree = self.grammar.sample()

    # Restrict possible configurations to control support and query lengths
    assert self.list_maxlength >= 9
    assert n_support >= 8

    # Partition lengths into disjoint support/query pools to prevent pure analogical
    # copying solutions. Support pool guarantees mix of short/long and even/odd lengths
    # to minimize rule induction ambiguity
    m = self.list_maxlength
    length_pool_support = [1, 2, 3, 4, m - 3, m - 2, m - 1, m]
    length_pool_query = np.arange(5, m - 3)

    lengths_query = self.rng.choice(length_pool_query, size=n_query)
    lengths_support = self.rng.permutation(np.resize(length_pool_support, n_support))

    def generate_example(
      length: int,
    ) -> tuple[jt.Int[np.ndarray, ' length_in'], jt.Int[np.ndarray, ' length_out']]:
      input = self.rng.integers(0, self.feature_maxval + 1, size=length)
      output = self.evaluate(input, parse_tree.expression)
      return (input, output)

    inputs_support, outputs_support = zip(
      *(generate_example(length) for length in lengths_support), strict=True
    )
    inputs_query, outputs_query = zip(
      *(generate_example(length) for length in lengths_query), strict=True
    )
    return ListFunctionTask(
      inputs_support, outputs_support, inputs_query, outputs_query, parse_tree
    )

  def evaluate(
    self, input_list: jt.Int[np.ndarray, ' input_length'], expression: str
  ) -> jt.Int[np.ndarray, ' output_length']:
    def insert_(
      lst: jt.Int[np.ndarray, ' length'], pos: int, elem: int
    ) -> jt.Int[np.ndarray, ' output_length']:
      if lst.size == 0:
        return lst
      if pos < 0:
        pos += len(lst)
      if pos < 0 or pos >= len(lst):
        return lst
      return np.insert(lst, pos, elem)

    def remove_(
      lst: jt.Int[np.ndarray, ' length'], pos: int
    ) -> jt.Int[np.ndarray, ' output_length']:
      if lst.size == 0:
        return lst
      if pos < 0:
        pos += len(lst)
      if pos < 0 or pos >= len(lst):
        return lst
      return np.concatenate((lst[:pos], lst[pos + 1 :]))

    def repeat_(
      lst: jt.Int[np.ndarray, ' length'], length: int
    ) -> jt.Int[np.ndarray, ' output_length']:
      if lst.size == 0:
        return lst
      return np.tile(lst, math.ceil(length / lst.size))[:length]

    def shift_(
      lst: jt.Int[np.ndarray, ' length'], offset: int
    ) -> jt.Int[np.ndarray, ' length']:
      # Shifts all elements by offset, wrapping the last element to the front
      return np.roll(lst, offset)

    def swap_(lst: jt.Int[np.ndarray, ' length']) -> jt.Int[np.ndarray, ' length']:
      mid = len(lst) // 2
      return np.concatenate((lst[mid:], lst[:mid]))

    def tail_(
      lst: jt.Int[np.ndarray, ' length'], length: int
    ) -> jt.Int[np.ndarray, ' output_length']:
      return lst[-length:]

    functions = {
      'insert_': insert_,
      'remove_': remove_,
      'repeat_': repeat_,
      'shift_': shift_,
      'swap_': swap_,
      'tail_': tail_,
    }

    return eval(expression, {'__builtins__': {}}, dict(x=input_list) | functions)


def format_as_prompt(task: ListFunctionTask) -> str:
  prompt_lines = []

  for i in range(len(task.inputs_support)):
    input_str = str(task.inputs_support[i].tolist())
    output_str = str(task.outputs_support[i].tolist())
    prompt_line = f'Example {i + 1}: {input_str} -> {output_str}'
    prompt_lines.append(prompt_line)

  prompt_lines.append('')

  for i in range(len(task.inputs_query)):
    input_str = str(task.inputs_query[i].tolist())
    prompt_line = f'Test {i + 1}: {input_str} -> ?'
    prompt_lines.append(prompt_line)

  return '\n'.join(prompt_lines)


class Shuffler:
  def __init__(self, seed: int | None) -> None:
    self.rng = np.random.default_rng(seed)

  def __call__(self, task: ListFunctionTask) -> ListFunctionTask:
    def _shuffle(inputs: tuple, outputs: tuple) -> tuple[tuple, tuple]:
      indices = self.rng.permutation(len(inputs))
      return (tuple(inputs[i] for i in indices), tuple(outputs[i] for i in indices))

    inputs_support, outputs_support = _shuffle(task.inputs_support, task.outputs_support)
    inputs_query, outputs_query = _shuffle(task.inputs_query, task.outputs_query)

    return ListFunctionTask(
      inputs_support=inputs_support,
      outputs_support=outputs_support,
      inputs_query=inputs_query,
      outputs_query=outputs_query,
      parse_tree=task.parse_tree,
    )


class Translator:
  def __init__(self, seed: int | None) -> None:
    self.rng = np.random.default_rng(seed)

  def __call__(
    self, task: ListFunctionTask, generator: ListFunctionGenerator, feature_maxval: int
  ) -> ListFunctionTask:
    vocab = np.arange(feature_maxval + 1)

    feature_map = self.rng.permutation(vocab)

    def apply_feature_map(x: np.ndarray) -> np.ndarray:
      return feature_map[x]

    inputs_support = tuple(apply_feature_map(x) for x in task.inputs_support)
    outputs_support = tuple(apply_feature_map(x) for x in task.outputs_support)
    inputs_query = tuple(apply_feature_map(x) for x in task.inputs_query)
    outputs_query = tuple(apply_feature_map(x) for x in task.outputs_query)

    def tree_mapping_fn(val: str, parent_symbol: Nonterminal) -> str:
      if parent_symbol.name == 'ELEM':
        return str(feature_map[int(val)])
      return val

    translated_parse_tree = task.parse_tree.map_literals(tree_mapping_fn)

    # Runtime verification of isomorphism
    for input_list, output_list in zip(inputs_support, outputs_support, strict=True):
      translated_out = generator.evaluate(input_list, translated_parse_tree.expression)
      assert np.array_equal(translated_out, output_list), (
        'Isomorphism verification failed on support set.'
      )

    for input_list, output_list in zip(inputs_query, outputs_query, strict=True):
      translated_out = generator.evaluate(input_list, translated_parse_tree.expression)
      assert np.array_equal(translated_out, output_list), (
        'Isomorphism verification failed on query set.'
      )

    return ListFunctionTask(
      inputs_support=inputs_support,
      outputs_support=outputs_support,
      inputs_query=inputs_query,
      outputs_query=outputs_query,
      parse_tree=translated_parse_tree,
    )


class ListintDataloader:
  def __init__(
    self,
    n_support: int,
    n_query: int,
    min_parse_tree_size: int,
    max_parse_tree_size: int,
    n_rules_per_size: int,
    n_translation_per_rule: int,
    n_shuffle_per_rule: int,
    seed: int,
    list_maxlength: int,
    feature_maxval: int,
  ) -> None:
    self.n_support = n_support
    self.n_query = n_query
    self.min_parse_tree_size = min_parse_tree_size
    self.max_parse_tree_size = max_parse_tree_size
    self.n_rules_per_size = n_rules_per_size
    self.n_translation_per_rule = n_translation_per_rule
    self.n_shuffle_per_rule = n_shuffle_per_rule
    self.feature_maxval = feature_maxval

    self.generator = ListFunctionGenerator(
      seed=seed, list_maxlength=list_maxlength, feature_maxval=feature_maxval
    )
    self.translator = Translator(seed=seed)
    self.shuffler = Shuffler(seed=seed)

  def package_batch(
    self,
    task: ListFunctionTask,
    set_id: str,
    variation_type: str,
    variation_idx: int,
  ) -> Batch:
    return Batch(
      prompt=format_as_prompt(task),
      target=np.array([str(x.tolist()) for x in task.outputs_query]),
      rule=np.array([task.parse_tree.expression] * self.n_query),
      info=dict(
        set_id=np.array([set_id] * self.n_query),
        variation_type=np.array([variation_type] * self.n_query),
        variation_idx=np.array([variation_idx] * self.n_query),
        parse_tree_prob=np.array([task.parse_tree.probability] * self.n_query),
        parse_tree_size=np.array([len(task.parse_tree)] * self.n_query),
      ),
    )

  def __iter__(self) -> Iterator[Batch]:
    seen_expressions = set()

    for size in range(self.min_parse_tree_size, self.max_parse_tree_size + 1):
      rules_yielded = 0
      for parse_tree in self.generator.grammar.sample_by_size(size):
        if rules_yielded >= self.n_rules_per_size:
          break

        if parse_tree.abstract_expression in seen_expressions:
          continue
        seen_expressions.add(parse_tree.abstract_expression)
        rules_yielded += 1

        task = self.generator.sample(self.n_support, self.n_query, parse_tree)
        set_id = hashlib.md5(parse_tree.abstract_expression.encode('utf-8')).hexdigest()

        yield self.package_batch(
          task=task, set_id=set_id, variation_type='canonical', variation_idx=0
        )

        for i in range(self.n_translation_per_rule):
          task_translated = self.translator(task, self.generator, self.feature_maxval)
          yield self.package_batch(
            task=task_translated,
            set_id=set_id,
            variation_type='translation',
            variation_idx=i + 1,
          )

        for i in range(self.n_shuffle_per_rule):
          task_shuffled = self.shuffler(task)
          yield self.package_batch(
            task=task_shuffled,
            set_id=set_id,
            variation_type='shuffle',
            variation_idx=i + 1,
          )


def create_listint_dataloader(
  n_support: int,
  n_query: int,
  min_parse_tree_size: int,
  max_parse_tree_size: int,
  n_rules_per_size: int,
  n_translation_per_rule: int,
  n_shuffle_per_rule: int,
  list_maxlength: int,
  feature_maxval: int,
  easy_instructions: bool,
  *,
  seed: int,
) -> tuple[ListintDataloader, str, type[pydantic.BaseModel]]:
  dataloader = ListintDataloader(
    n_support=n_support,
    n_query=n_query,
    min_parse_tree_size=min_parse_tree_size,
    max_parse_tree_size=max_parse_tree_size,
    n_rules_per_size=n_rules_per_size,
    n_translation_per_rule=n_translation_per_rule,
    n_shuffle_per_rule=n_shuffle_per_rule,
    seed=seed,
    list_maxlength=list_maxlength,
    feature_maxval=feature_maxval,
  )
  operations_description = OPERATIONS_DESCRIPTION if easy_instructions else '\n'
  system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
    n_support=n_support,
    n_query=n_query,
    operations_description=operations_description,
  )
  schema = pydantic.create_model(
    'Answer', **{str(i + 1): (str, ...) for i in range(n_query)}
  )

  return (dataloader, system_prompt, schema)
