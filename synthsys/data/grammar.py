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

"""Probabilistic context-free grammar (PCFG) engine and parse tree abstractions."""

import dataclasses
import math
import random
from collections import defaultdict
from typing import Callable
from typing import Iterator
from typing import Type
from typing import TypeVar

T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class Nonterminal:
  name: str
  allow_resample: bool = False

  def __repr__(self) -> str:
    return self.name if not self.allow_resample else f'{self.name}?'


@dataclasses.dataclass(frozen=True)
class LiteralNode[T]:
  name: str
  type: Type[T]

  def __repr__(self) -> str:
    return f'{self.name}: {self.type.__name__}'

  def value(self) -> T:
    return self.type(self.name)


@dataclasses.dataclass(frozen=True)
class VariableNode:
  name: str
  type: Type

  def __repr__(self) -> str:
    return f'{self.name}: {self.type.__name__}'


Terminal = LiteralNode | VariableNode


@dataclasses.dataclass(frozen=True)
class Production:
  lhs: Nonterminal
  rhs: tuple[Nonterminal | VariableNode | LiteralNode, ...]
  probability: float
  fun_name: str | None = None


@dataclasses.dataclass(frozen=True)
class ParseTree:
  symbol: Nonterminal | VariableNode | LiteralNode
  children: tuple['ParseTree', ...] | None
  production_used: Production | None

  @property
  def probability(self) -> float:
    if self.children is None:
      return 1.0

    child_prob = math.prod(child.probability for child in self.children)
    production_prob = self.production_used.probability if self.production_used else 1.0
    return production_prob * child_prob

  @property
  def expression(self) -> str:
    if self.children is None:
      return self.symbol.name

    if self.production_used and self.production_used.fun_name:
      fun_args = ', '.join(c.expression for c in self.children)
      return f'{self.production_used.fun_name}({fun_args})'

    return self.children[0].expression

  @property
  def abstract_expression(self) -> str:
    if self.children is None:
      if isinstance(self.symbol, LiteralNode):
        return 'LITERAL'
      return self.symbol.name

    if self.production_used and self.production_used.fun_name:
      fun_args = ', '.join(c.abstract_expression for c in self.children)
      return f'{self.production_used.fun_name}({fun_args})'

    return self.children[0].abstract_expression

  def __repr__(self) -> str:
    def _tree_repr(node: 'ParseTree', prefix: str = '', is_last: bool = True) -> str:
      # Build the node label
      node_label = str(node.symbol)
      if node.production_used and node.production_used.fun_name:
        node_label = f'{node.symbol} -> {node.production_used.fun_name}'

      result = (
        f'{prefix}{"└── " if is_last else "├── "}{node_label} '
        f'(p={node.probability:.3f})\n'
      )

      if node.children is not None:
        for i, child in enumerate(node.children):
          child_prefix = prefix + ('    ' if is_last else '│   ')
          result += _tree_repr(child, child_prefix, i == len(node.children) - 1)

      return result

    return _tree_repr(self).rstrip()

  def __len__(self) -> int:
    """Number of nodes in the ParseTree."""
    count = 1

    if self.children is not None:
      for child in self.children:
        count += len(child)

    return count

  def is_complete(self) -> bool:
    """Check that a parse tree has no unexpanded nonterminals."""
    if self.children is None:
      return not isinstance(self.symbol, Nonterminal)
    return all(child.is_complete() for child in self.children)

  def map_literals(
    self,
    mapping_fn: Callable[[str, Nonterminal], str],
    parent_symbol: Nonterminal | None = None,
  ) -> 'ParseTree':
    if self.children is None:
      if isinstance(self.symbol, LiteralNode) and parent_symbol is not None:
        new_name = mapping_fn(self.symbol.name, parent_symbol)
        new_symbol = LiteralNode(new_name, self.symbol.type)
        return ParseTree(new_symbol, None, None)
      return self

    new_parent_symbol = (
      self.symbol if isinstance(self.symbol, Nonterminal) else parent_symbol
    )
    new_children = tuple(
      child.map_literals(mapping_fn, new_parent_symbol) for child in self.children
    )
    return ParseTree(self.symbol, new_children, self.production_used)


class ProbabilisticContextFreeGrammar:
  """Define a probabilistic context-free grammar over nested functions."""

  def __init__(
    self,
    productions: list[Production],
    start_symbol: Nonterminal,
    seed: int,
    max_parse_tree_size: int = 100,
  ) -> None:
    self.start_symbol = start_symbol
    self.productions: defaultdict[Nonterminal, list[Production]] = defaultdict(list)
    for production in productions:
      self.productions[production.lhs].append(production)

    self.max_parse_tree_size = max_parse_tree_size
    self.counts = self._compute_size_counts(max_size=max_parse_tree_size)
    self.rng = random.Random(seed)

    for nonterminal, prods in self.productions.items():
      total_prob = sum(prod.probability for prod in prods)
      if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(
          f"Probabilities for nonterminal '{nonterminal.name}' sum to {total_prob} != 1.0"
        )

    if self.is_finite():
      raise ValueError('Grammar only produces fintely many parse trees.')

  def is_finite(self) -> bool:
    """Check whether the cfg produces finitely many parse trees."""
    # Build a directed graph of nonterminal-to-nonterminal dependencies
    graph: dict[Nonterminal, list[Nonterminal]] = defaultdict(list)

    for lhs, prods in self.productions.items():
      for prod in prods:
        if prod.probability > 0:
          for node in prod.rhs:
            if isinstance(node, Nonterminal):
              graph[lhs].append(node)

    # Check for cycles using DFS with 0: not visited, 1: currently visiting, 2: finished
    processed: dict[Nonterminal, int] = defaultdict(int)

    def has_cycle(node: Nonterminal) -> bool:
      match processed[node]:
        case 0:
          processed[node] = 1
        case 1:
          return True
        case 2:
          return False

      for neighbor in graph[node]:
        if has_cycle(neighbor):
          return True

      processed[node] = 2
      return False

    for nonterminal in self.productions.keys():
      if has_cycle(nonterminal):
        return False

    return True

  def sample(
    self, symbol: Nonterminal | VariableNode | LiteralNode | None = None
  ) -> ParseTree:
    """Sample a derivation from the PCFG."""
    if symbol is None:
      symbol = self.start_symbol

    match symbol:
      case VariableNode() | LiteralNode():
        return ParseTree(symbol, children=None, production_used=None)

      case Nonterminal():
        possible_productions = self.productions[symbol]
        probabilities = [prod.probability for prod in possible_productions]
        [production] = self.rng.choices(possible_productions, weights=probabilities)

        children = tuple(self.sample(rhs_symbol) for rhs_symbol in production.rhs)
        return ParseTree(symbol, children=children, production_used=production)

      case _:
        raise ValueError(f'Unknown symbol type: {type(symbol)}')

  def resample(self, parse_tree: ParseTree) -> ParseTree:
    """Resample subtrees of Nonterminal nodes marked with allow_resample."""
    if parse_tree.children is None:
      return parse_tree

    if isinstance(parse_tree.symbol, Nonterminal) and parse_tree.symbol.allow_resample:
      return self.sample(parse_tree.symbol)

    new_children = tuple(self.resample(child) for child in parse_tree.children)

    return ParseTree(parse_tree.symbol, new_children, parse_tree.production_used)

  def sample_by_size(self, parse_tree_size: int) -> Iterator[ParseTree]:
    if parse_tree_size > self.max_parse_tree_size:
      raise ValueError(
        'Requested `parse_tree_size` larger than the `max_parse_tree_size`'
        'specified at initialization.'
      )
    count_at_size = self.counts[self.start_symbol].get(parse_tree_size, 0)
    for index in self.rng.sample(range(count_at_size), k=count_at_size):
      yield self._build_tree_at_index(
        self.start_symbol, parse_tree_size, index, self.counts
      )

  def enumerate(self) -> Iterator[ParseTree]:
    for size in range(1, self.max_parse_tree_size + 1):
      count_at_size = self.counts[self.start_symbol].get(size, 0)

      for index in range(count_at_size):
        yield self._build_tree_at_index(self.start_symbol, size, index, self.counts)

  def _compute_size_counts(self, max_size: int) -> dict[Nonterminal, dict[int, int]]:
    counts: dict[Nonterminal, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    # Initialize with counts for terminal productions
    for productions in self.productions.values():
      for prod in productions:
        if len(prod.rhs) == 1 and isinstance(prod.rhs[0], Terminal):
          # Terminals create trees of size 2: start_symbol->terminal
          counts[prod.lhs][2] += 1

    # Build up counts for larger sizes (DP)
    for size in range(3, max_size + 1):
      for productions in self.productions.values():
        for prod in productions:
          if len(prod.rhs) == 1 and isinstance(prod.rhs[0], Nonterminal):
            counts[prod.lhs][size] += counts[prod.rhs[0]].get(size - 1, 0)
          elif len(prod.rhs) > 1:
            counts[prod.lhs][size] += self._count_partitions(
              prod.rhs, 0, size - 1, counts
            )
          else:
            pass

    return counts

  def _count_partitions(
    self,
    args: tuple[Nonterminal | VariableNode | LiteralNode, ...],
    start_idx: int,
    size_left: int,
    counts: dict[Nonterminal, dict[int, int]],
  ) -> int:
    """Count ways to partition size_left among args[start_idx:]."""
    if start_idx == len(args):
      return 1 if size_left == 0 else 0

    if size_left < len(args) - start_idx:
      return 0  # Not enough size for remaining arguments

    total = 0
    for arg_size in range(1, size_left + 1):
      arg_count = self._get_symbol_count(args[start_idx], arg_size, counts)
      if arg_count > 0:
        total += arg_count * self._count_partitions(
          args, start_idx + 1, size_left - arg_size, counts
        )

    return total

  def _get_symbol_count(
    self,
    symbol: Nonterminal | VariableNode | LiteralNode,
    size: int,
    counts: dict[Nonterminal, dict[int, int]],
  ) -> int:
    """Get the number of trees of given size for a symbol."""
    if isinstance(symbol, Nonterminal):
      return counts[symbol].get(size, 0)
    else:
      return 1 if size == 1 else 0

  def _build_tree_at_index(
    self,
    symbol: Nonterminal | VariableNode | LiteralNode,
    size: int,
    index: int,
    counts: dict[Nonterminal, dict[int, int]],
  ) -> ParseTree:
    """Build the index-th tree of given size for a symbol using FEAT algorithm."""

    if isinstance(symbol, Terminal):
      assert size == 1 and index == 0
      return ParseTree(symbol, None, None)

    accumulated = 0
    for prod in self.productions[symbol]:
      if len(prod.rhs) == 1:
        match prod.rhs[0]:
          case LiteralNode() | VariableNode():
            count = 1 if size == 2 else 0
          case Nonterminal():
            count = counts[prod.rhs[0]].get(size - 1, 0) if size > 1 else 0
      else:
        # Multiple symbols on rhs, need to count possible partitions
        count = self._count_partitions(prod.rhs, 0, size - 1, counts)

      if accumulated + count > index:
        # This production contains our target index
        local_index = index - accumulated

        if len(prod.rhs) == 1:
          chld = self._build_tree_at_index(prod.rhs[0], size - 1, local_index, counts)
          return ParseTree(symbol, (chld,), prod)
        else:
          # Need to determine how to distribute size among multiple children
          children = []
          remaining_index = local_index
          size_left = size - 1

          for arg_idx, arg in enumerate(prod.rhs):
            # Try each possible size for this argument
            for arg_size in range(1, size_left + 1):
              arg_count = self._get_symbol_count(arg, arg_size, counts)
              if arg_count == 0:
                continue

              # Count ways to partition remaining size among remaining args
              rest_count = self._count_partitions(
                prod.rhs, arg_idx + 1, size_left - arg_size, counts
              )
              partition_count = arg_count * rest_count

              if remaining_index < partition_count:
                # This arg_size contains our index
                child_index = remaining_index // rest_count
                remaining_index = remaining_index % rest_count
                children.append(
                  self._build_tree_at_index(arg, arg_size, child_index, counts)
                )
                size_left -= arg_size
                break

              remaining_index -= partition_count
            else:
              raise ValueError(f'Index {index} out of range for production {prod}')

          return ParseTree(symbol, tuple(children), prod)

      accumulated += count

    raise IndexError(f'Could not find index {index} at symbol {symbol} for size {size}')
