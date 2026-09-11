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

"""Data generation for mlc rule induction tasks."""

import hashlib
import itertools
import random
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
from typing import Literal
from typing import Protocol

import numpy as np
import pydantic

from synthsys.data.base import Batch


class Vocabularies(Enum):
  MLC = (
    ('dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer'),
    ('RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK'),
  )
  LETTERS = (tuple(string.ascii_lowercase), tuple(string.ascii_uppercase))
  NUMBERS = (
    tuple(map(str, range(100))),
    tuple(map(str, range(100, 200))),
  )


SYSTEM_PROMPT_TEMPLATE = """
Your task is to discover the grammatical rules of an artificial language.
In the following, you will be presented with examples of input and output string pairs.
These pairs demonstrate how sequences of primitive tokens are transformed by grammatical rules.
Afterwards, you will be presented with {n_query} new input sequence(s), and you have to translate them into output sequences based on the grammatical rules you have inferred.
Try to identify the simplest deterministic rules that govern these transformations and apply them to the new input sequences.

Every artificial language consists of:
1. Primitive rules: Simple one-to-one mappings from an input token to an output token.
2. Function rules: These operate on one or two arguments. Functions are represented by specific tokens and apply to their adjacent arguments.
  - A 1-argument function appears after its argument (e.g., "arg1 func"). A 2-argument function appears between its arguments (e.g., "arg1 func arg2").
  - Each argument is either a single primitive token or a sequence of tokens.
  - A function transforms its evaluated arguments into a new sequence by duplicating and/or rearranging them.
  - When the same 2-argument function is chained with itself, it evaluates left-to-right (e.g., "arg1 func arg2 func arg3" is evaluated as "[arg1 func arg2] func arg3") otherwise functions that accept sequences follow a strict priority order: If Rule A evaluates before Rule B, and Rule B before Rule C, then Rule A always evaluates before Rule C.
"""  # noqa: E501


ArgType = Literal['primitive', 'string']


@dataclass
class LearningEpisode:
  support_in: list[str]
  support_out: list[str]
  query_in: list[str]
  query_out: list[str]


@dataclass
class Rule:
  name: str
  arg_types: list[ArgType]
  in_template: str
  out_template: str


class SyntaxTreeNode(Protocol):
  def render_input(self) -> str: ...
  def render_output(self, primitives_map: dict[str, str]) -> str: ...


class PrimitiveNode:
  def __init__(self, in_str: str) -> None:
    self.in_str = in_str

  def render_input(self) -> str:
    return self.in_str

  def render_output(self, primitives_map: dict[str, str]) -> str:
    return primitives_map[self.in_str]


class RuleNode:
  def __init__(self, rule: Rule, children: list[SyntaxTreeNode]) -> None:
    self.rule = rule
    self.children = children

  def render_input(self) -> str:
    rendered_children = [c.render_input() for c in self.children]
    return self.rule.in_template.format(*rendered_children)

  def render_output(self, primitives_map: dict[str, str]) -> str:
    rendered_children = [c.render_output(primitives_map) for c in self.children]
    return self.rule.out_template.format(*rendered_children)


class Grammar:
  def __init__(
    self,
    primitives: list[str],
    rules: list[Rule],
    seed: int | None = None,
  ) -> None:
    self.primitives = primitives
    self.concat_rule = Rule(
      name='concat',
      arg_types=['primitive', 'string'],
      in_template='{0} {1}',
      out_template='{0} {1}',
    )
    self.rules = [*rules, self.concat_rule]
    self.rng = random.Random(seed)

  def sample_primitive(self) -> PrimitiveNode:
    return PrimitiveNode(self.rng.choice(self.primitives))

  def sample_syntax_tree(self, max_depth: int) -> SyntaxTreeNode:
    if max_depth <= 1:
      return self.sample_primitive()

    options = ['primitive', *self.rules]
    choice = self.rng.choice(options)

    match choice:
      case 'primitive':
        return self.sample_primitive()
      case Rule() as rule:
        children = []
        for arg_type in rule.arg_types:
          match arg_type:
            case 'primitive':
              children.append(self.sample_primitive())
            case 'string':
              children.append(self.sample_syntax_tree(max_depth - 1))
        return RuleNode(rule, children)
      case _:
        raise RuntimeError(f'Unexpected AST node choice: {choice}')


class Parser:
  """Deterministic regex interpreter (semantics)"""

  def __init__(self, primitives: list[tuple[str, str]], rules: list[Rule]) -> None:
    self.primitives_map = {p[0]: p[1] for p in primitives}
    self.rules = rules

    prim_pattern = r'(' + '|'.join(self.primitives_map.keys()) + r')'
    str_pattern = r'([ a-zA-Z0-9]+)'

    self.compiled_rules = []
    for rule in self.rules:
      regex = rule.in_template
      for i, arg_type in enumerate(rule.arg_types):
        pattern = prim_pattern if arg_type == 'primitive' else str_pattern
        regex = regex.replace(f'{{{i}}}', pattern)

      self.compiled_rules.append((re.compile(regex), rule))

  def parse(self, s: str, depth: int = 0) -> str:
    s = s.strip()

    # Base case: string is a known primitive
    if s in self.primitives_map:
      return self.primitives_map[s]

    # Evaluate rules in priority order
    for regex, rule in self.compiled_rules:
      match = regex.fullmatch(s)
      if match:
        groups = match.groups()
        out_str = rule.out_template
        rendered_args = [self.parse(g, depth + 1) for g in groups]
        for i, arg in enumerate(rendered_args):
          out_str = out_str.replace(f'{{{i}}}', arg)
          if len(out_str) > 1000:
            raise RecursionError('Output string exceeded 1000 characters.')
        return out_str

    return s


class Language:
  """Artificial language created from a Grammar and Parser."""

  def __init__(
    self,
    primitives: list[tuple[str, str]],
    rules: list[Rule],
    max_example_length: int,
    seed: int | None = None,
  ) -> None:
    self.primitives = primitives
    self.rules = rules
    self.max_example_length = max_example_length
    self.rng = random.Random(seed)

    self.grammar = Grammar(
      primitives=[p_in for p_in, p_out in primitives],
      rules=self.rules,
      seed=seed,
    )

    self.parser = Parser(
      primitives=primitives,
      rules=self.grammar.rules,
    )

    self.symbols = {p[0] for p in self.primitives} | {r.name for r in self.grammar.rules}

  @property
  def hash_signature(self) -> str:
    """
    Creates a unique, order-invariant string signature for the structural logic of the
    grammar rules. This ensures we do not sample the same language twice.
    """
    rule_strings = []
    for rule in self.rules:
      arg_str = ' '.join(rule.arg_types)
      abstract_in_template = rule.in_template.replace(rule.name, 'FUNC')
      rule_strings.append(f'{abstract_in_template} -> {rule.out_template} ({arg_str})')
    rule_strings.sort()
    return '\n'.join(rule_strings)

  def generate_lexical_anchors(self) -> list[tuple[str, str]]:
    """Demonstrate the 1-to-1 mapping of primitive tokens."""
    anchors = []
    for p_in, _ in self.primitives:
      rule = self.rng.choice(self.rules)
      p_node = PrimitiveNode(p_in)
      if len(rule.arg_types) == 1:
        tree = RuleNode(rule, [p_node])
      else:
        tree = RuleNode(rule, [p_node, p_node])
      in_str = tree.render_input()
      out_str = self.parser.parse(in_str)
      anchors.append((in_str, out_str))
    return anchors

  def generate_template_resolvers(self) -> list[tuple[str, str]]:
    """Demonstrate argument arity, argument types and output templates for each rule."""
    resolvers = []
    for rule in self.rules:
      p1, p2, p3 = self.rng.sample(self.primitives, 3)
      match tuple(rule.arg_types):
        case ('string', 'string'):
          c1 = RuleNode(
            self.grammar.concat_rule, [PrimitiveNode(p1[0]), PrimitiveNode(p2[0])]
          )
          c2 = RuleNode(
            self.grammar.concat_rule, [PrimitiveNode(p2[0]), PrimitiveNode(p3[0])]
          )
          tree = RuleNode(rule, [c1, c2])
        case ('string', 'primitive'):
          c1 = RuleNode(
            self.grammar.concat_rule, [PrimitiveNode(p1[0]), PrimitiveNode(p2[0])]
          )
          tree = RuleNode(rule, [c1, PrimitiveNode(p3[0])])
        case ('primitive', 'string'):
          c2 = RuleNode(
            self.grammar.concat_rule, [PrimitiveNode(p1[0]), PrimitiveNode(p2[0])]
          )
          tree = RuleNode(rule, [PrimitiveNode(p3[0]), c2])
        case ('primitive', 'primitive'):
          tree = RuleNode(rule, [PrimitiveNode(p1[0]), PrimitiveNode(p2[0])])
        case ('string',):
          c1 = RuleNode(
            self.grammar.concat_rule, [PrimitiveNode(p1[0]), PrimitiveNode(p2[0])]
          )
          tree = RuleNode(rule, [c1])
        case ('primitive',):
          tree = RuleNode(rule, [PrimitiveNode(p1[0])])
        case _:
          raise ValueError(f'Unexpected argument types: {rule.arg_types}')

      in_str = tree.render_input()
      out_str = self.parser.parse(in_str)
      resolvers.append((in_str, out_str))

    return resolvers

  def _generate_nested_rules(
    self, rule_outer: Rule, rule_inner: Rule
  ) -> list[tuple[str, str]]:
    # Ensure we respect parser constraints where higher priority rule must be the root
    assert self.rules.index(rule_outer) <= self.rules.index(rule_inner)
    inner_prims = self.rng.sample(self.primitives, len(rule_inner.arg_types))
    inner_node = RuleNode(rule_inner, [PrimitiveNode(p[0]) for p in inner_prims])

    outer_nodes = []
    match tuple(rule_outer.arg_types):
      case ('string',):
        outer_nodes.append(RuleNode(rule_outer, [inner_node]))
      case ('string', 'primitive'):
        prim = self.rng.choice(self.primitives)
        outer_nodes.append(RuleNode(rule_outer, [inner_node, PrimitiveNode(prim[0])]))
      case ('primitive', 'string'):
        prim = self.rng.choice(self.primitives)
        outer_nodes.append(RuleNode(rule_outer, [PrimitiveNode(prim[0]), inner_node]))
      case ('string', 'string'):
        prim1 = self.rng.choice(self.primitives)
        outer_nodes.append(RuleNode(rule_outer, [inner_node, PrimitiveNode(prim1[0])]))
        prim2 = self.rng.choice(self.primitives)
        outer_nodes.append(RuleNode(rule_outer, [PrimitiveNode(prim2[0]), inner_node]))
      case _:
        raise ValueError(f'Outer rule must accept a string: {rule_outer.arg_types}')

    return [
      (node.render_input(), self.parser.parse(node.render_input()))
      for node in outer_nodes
    ]

  def generate_precedence_pattern(self) -> list[tuple[str, str]]:
    """Demonstrate rule priority order."""
    string_rules = [r for r in self.rules if 'string' in r.arg_types]
    return [
      proof
      for rule_a, rule_b in itertools.combinations(string_rules, 2)
      for proof in self._generate_nested_rules(rule_outer=rule_a, rule_inner=rule_b)
    ]

  def sample_unambiguous_tree(self, max_depth: int, max_tries: int) -> SyntaxTreeNode:
    for _ in range(max_tries):
      tree = self.grammar.sample_syntax_tree(max_depth)
      in_str = tree.render_input()
      try:
        deterministic_out = self.parser.parse(in_str)
        tree_out = tree.render_output(self.parser.primitives_map)
      except RecursionError:
        continue
      if deterministic_out == tree_out:
        return tree

    raise RuntimeError(f'Max tries ({max_tries}) exceeded while sampling trees.')

  def generate_entanglers(
    self, n_needed: int, exclude: set[tuple[str, str]], max_depth: int, max_tries: int
  ) -> list[tuple[str, str]]:
    """Generate nested compositions of rules."""
    entanglers = []
    seen = set(exclude)
    tries = 0
    while len(entanglers) < n_needed and tries < max_tries:
      tries += 1
      tree = self.sample_unambiguous_tree(max_depth, max_tries)

      in_str = tree.render_input()
      words = in_str.split()
      func_count = sum(1 for w in words if w in {r.name for r in self.rules})
      if func_count < 2:
        continue

      if len(words) > self.max_example_length:
        # input string too long
        continue

      out_str = tree.render_output(self.parser.primitives_map)

      if len(out_str.split()) > self.max_example_length:
        # output string too long
        continue

      example = (in_str, out_str)
      if example in seen:
        continue

      seen.add(example)
      entanglers.append(example)

    if len(entanglers) < n_needed:
      raise RuntimeError(f'Max tries ({max_tries}) exceeded while generating entanglers.')

    return entanglers

  def generate_episode(
    self, n_support: int, n_query: int, max_depth: int, max_tries: int
  ) -> LearningEpisode:
    """Creates support and query examples from the language.

    Support examples are created using a curriculum that aims to minimize ambiguity
    without relying on simplistic, depth-0 "dictionary" examples. It needs to handle

    1. Lexical resolution: Determine the 1-to-1 mapping of primitive tokens.
    2. Structural resolution: Determine argument arity (1 or 2 arguments), argument types
      (string/primitive) and  output template (reorder/duplicate) for each rule.
    3. Precedence resolution: Determine the priority order of rules.
    """

    # Generate the curriculum examples
    anchors = self.generate_lexical_anchors()
    resolvers = self.generate_template_resolvers()
    proofs = self.generate_precedence_pattern()
    curriculum = set(anchors + resolvers + proofs)

    entanglers = self.generate_entanglers(
      n_support + n_query, curriculum, max_depth, max_tries
    )

    # Split entangler examples into support and query set
    support_set = list(curriculum) + entanglers[:n_support]
    query_set = entanglers[n_support:]

    self.rng.shuffle(support_set)
    self.rng.shuffle(query_set)

    return LearningEpisode(
      support_in=[e[0] for e in support_set],
      support_out=[e[1] for e in support_set],
      query_in=[e[0] for e in query_set],
      query_out=[e[1] for e in query_set],
    )


class MetaGrammar:
  def __init__(
    self,
    n_primitives: int,
    n_rules: int,
    input_vocab: tuple[str, ...],
    output_vocab: tuple[str, ...],
    max_rhs_length: int,
    max_example_length: int,
    p_lhs_onearg: float,
    seed: int | None,
  ) -> None:
    assert n_primitives > 2
    assert n_rules > 0

    self.n_primitives = n_primitives
    self.n_rules = n_rules
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.max_rhs_length = max_rhs_length
    self.max_example_length = max_example_length
    self.p_lhs_onearg = p_lhs_onearg
    self.rng = random.Random(seed)
    self.tabu_set: set[str] = set()

  def sample(self, max_tries: int = 100) -> Language:
    for _ in range(max_tries):
      input_vocab = list(self.input_vocab)
      output_vocab = list(self.output_vocab)
      self.rng.shuffle(input_vocab)
      self.rng.shuffle(output_vocab)

      primitives = list(
        zip(
          input_vocab[: self.n_primitives],
          output_vocab[: self.n_primitives],
          strict=True,
        )
      )

      func_words = input_vocab[self.n_primitives : self.n_primitives + self.n_rules]
      rules = []

      for func_name in func_words:
        # Sample left-hand side (LHS)
        is_one_arg = self.rng.random() < self.p_lhs_onearg
        n_args = 1 if is_one_arg else 2
        arg_types = self.rng.choices(['primitive', 'string'], k=n_args)
        in_template = f'{{0}} {func_name}' if is_one_arg else f'{{0}} {func_name} {{1}}'

        # Sample right-hand side (RHS) ensuring each lhs arg appears at least once
        target_length = self.rng.randint(2, self.max_rhs_length)
        rhs_indices = list(range(n_args)) + self.rng.choices(
          range(n_args), k=target_length - n_args
        )
        self.rng.shuffle(rhs_indices)
        out_template = ' '.join([f'{{{idx}}}' for idx in rhs_indices])

        rules.append(
          Rule(
            name=func_name,
            arg_types=arg_types,
            in_template=in_template,
            out_template=out_template,
          )
        )

      # Ensure language is compositional (at least one rule accepts a string)
      if not any('string' in rule.arg_types for rule in rules):
        continue

      language = Language(
        primitives=primitives,
        rules=rules,
        max_example_length=self.max_example_length,
        seed=self.rng.randint(0, 2**32 - 1),
      )

      # Ensure language is structurally novel
      signature = language.hash_signature
      if signature in self.tabu_set:
        continue

      self.tabu_set.add(signature)
      return language

    raise RuntimeError(
      f'Failed to generate a novel language not in tabu list after {max_tries} tries.'
    )


class Translator:
  def __init__(self, seed: int | None) -> None:
    self.rng = random.Random(seed)

  def __call__(
    self,
    episode: LearningEpisode,
    input_symbols_old: tuple[str, ...],
    input_symbols_new: tuple[str, ...],
    output_symbols_old: tuple[str, ...],
    output_symbols_new: tuple[str, ...],
  ) -> LearningEpisode:
    """
    Sample a random mapping between old and new vocabulary and use it to translate the
    LearningEpisode.
    """
    if len(input_symbols_new) < len(input_symbols_old):
      raise ValueError('Not enough new input symbols to map all old input symbols.')
    if len(output_symbols_new) < len(output_symbols_old):
      raise ValueError('Not enough new output symbols to map all old output symbols.')

    # Sample without replacement to create a 1-to-1 randomized mapping
    in_map = dict(
      zip(
        input_symbols_old,
        self.rng.sample(input_symbols_new, len(input_symbols_old)),
        strict=True,
      )
    )
    out_map = dict(
      zip(
        output_symbols_old,
        self.rng.sample(output_symbols_new, len(output_symbols_old)),
        strict=True,
      )
    )

    def translate(sentences: list[str], vocab_map: dict[str, str]) -> list[str]:
      return [
        ' '.join(vocab_map[word] for word in sentence.split()) for sentence in sentences
      ]

    return LearningEpisode(
      support_in=translate(episode.support_in, in_map),
      support_out=translate(episode.support_out, out_map),
      query_in=translate(episode.query_in, in_map),
      query_out=translate(episode.query_out, out_map),
    )


class Recomposer:
  def __init__(self, seed: int | None) -> None:
    self.rng = random.Random(seed)

  def __call__(self, in_str: str, primitives: list[str]) -> str:
    primitives_set = set(primitives)
    tokens = in_str.split()

    # Identify all primitives in the string and permute them
    prim_indices = [i for i, t in enumerate(tokens) if t in primitives_set]
    prim_tokens = [tokens[i] for i in prim_indices]
    self.rng.shuffle(prim_tokens)

    # Re-inject permuted primitives back into their original positions
    for i, new_prim in zip(prim_indices, prim_tokens, strict=True):
      tokens[i] = new_prim

    return ' '.join(tokens)


class Shuffler:
  def __init__(self, seed: int | None) -> None:
    self.rng = random.Random(seed)

  def __call__(self, episode: LearningEpisode) -> LearningEpisode:
    def shuffle_list_pair(
      list_a: list[str], list_b: list[str]
    ) -> tuple[list[str], list[str]]:
      pairs = list(zip(list_a, list_b, strict=True))
      self.rng.shuffle(pairs)
      return [p[0] for p in pairs], [p[1] for p in pairs]

    support_in, support_out = shuffle_list_pair(episode.support_in, episode.support_out)
    query_in, query_out = shuffle_list_pair(episode.query_in, episode.query_out)

    return LearningEpisode(
      support_in=support_in,
      support_out=support_out,
      query_in=query_in,
      query_out=query_out,
    )


def format_as_prompt(episode: LearningEpisode) -> str:
  prompt_lines = []

  for i in range(len(episode.support_in)):
    prompt_lines.append(
      f'Example {i + 1}: IN: {episode.support_in[i]}  -->  OUT: {episode.support_out[i]}'
    )

  prompt_lines.append('')

  for i in range(len(episode.query_in)):
    prompt_lines.append(f'Test {i + 1}: IN: {episode.query_in[i]}  -->  OUT: ?')

  return '\n'.join(prompt_lines)


class MLCDataloader:
  """
  MLC task generation

  - MetaGrammar: Generates the vocabulary and rules of an artificial language.
  - Grammar (Syntax): Generates structurally valid input sequences .
  - Parser (Semantics): Evaluates utterances to determine their meaning.
  - Language: Orchestrates the grammar and parser to sample consistent dataset episodes.
  - Translator: Replaces vocabulary of a LearningEpisode but keeps structure unchanged.
  """

  def __init__(
    self,
    n_primitives: int,
    n_rules: int,
    vocabulary: Literal['MLC', 'LETTERS', 'NUMBERS'],
    max_rhs_length: int,
    max_example_length: int,
    p_lhs_onearg: float,
    n_languages: int,
    n_translation_per_language: int,
    n_recomposition_per_language: int,
    n_shuffle_per_language: int,
    n_support: int,
    n_query: int,
    seed: int,
  ) -> None:
    self.n_languages = n_languages
    self.n_translation_per_language = n_translation_per_language
    self.n_recomposition_per_language = n_recomposition_per_language
    self.n_shuffle_per_language = n_shuffle_per_language
    self.n_support = n_support
    self.n_query = n_query

    self.input_vocab, self.output_vocab = Vocabularies[vocabulary].value

    self.meta_grammar = MetaGrammar(
      n_primitives=n_primitives,
      n_rules=n_rules,
      input_vocab=self.input_vocab,
      output_vocab=self.output_vocab,
      max_rhs_length=max_rhs_length,
      max_example_length=max_example_length,
      p_lhs_onearg=p_lhs_onearg,
      seed=seed,
    )
    self.translator = Translator(seed=seed)
    self.recomposer = Recomposer(seed=seed)
    self.shuffler = Shuffler(seed=seed)

  def __iter__(self) -> Iterator[Batch]:
    for _ in range(self.n_languages):
      language = self.meta_grammar.sample()
      episode = language.generate_episode(
        self.n_support, self.n_query, max_depth=16, max_tries=10000
      )
      set_id = hashlib.md5(language.hash_signature.encode('utf-8')).hexdigest()

      yield Batch(
        prompt=format_as_prompt(episode),
        target=np.array(episode.query_out),
        rule=np.array([language.hash_signature] * self.n_query),
        info=dict(
          set_id=np.array([set_id] * self.n_query),
          variation_type=np.array(['canonical'] * self.n_query),
          variation_idx=np.array([0] * self.n_query),
        ),
      )

      # Create lexical/surface variations
      for i in range(self.n_translation_per_language):
        translated_episode = self.translator(
          episode,
          input_symbols_old=self.input_vocab,
          input_symbols_new=self.input_vocab,
          output_symbols_old=self.output_vocab,
          output_symbols_new=self.output_vocab,
        )
        yield Batch(
          prompt=format_as_prompt(translated_episode),
          target=np.array(translated_episode.query_out),
          rule=np.array([language.hash_signature] * self.n_query),
          info=dict(
            set_id=np.array([set_id] * self.n_query),
            variation_type=np.array(['translation'] * self.n_query),
            variation_idx=np.array([i + 1] * self.n_query),
          ),
        )

      # Create compositional variations
      for i in range(self.n_recomposition_per_language):
        active_primitives = [p[0] for p in language.primitives]
        new_query_in = [self.recomposer(q, active_primitives) for q in episode.query_in]
        new_query_out = [language.parser.parse(q) for q in new_query_in]
        recomposed_episode = LearningEpisode(
          support_in=episode.support_in,
          support_out=episode.support_out,
          query_in=new_query_in,
          query_out=new_query_out,
        )
        yield Batch(
          prompt=format_as_prompt(recomposed_episode),
          target=np.array(recomposed_episode.query_out),
          rule=np.array([language.hash_signature] * self.n_query),
          info=dict(
            set_id=np.array([set_id] * self.n_query),
            variation_type=np.array(['recomposition'] * self.n_query),
            variation_idx=np.array([i + 1] * self.n_query),
          ),
        )

      # Create shuffle variations
      for i in range(self.n_shuffle_per_language):
        shuffled_episode = self.shuffler(episode)
        yield Batch(
          prompt=format_as_prompt(shuffled_episode),
          target=np.array(shuffled_episode.query_out),
          rule=np.array([language.hash_signature] * self.n_query),
          info=dict(
            set_id=np.array([set_id] * self.n_query),
            variation_type=np.array(['shuffle'] * self.n_query),
            variation_idx=np.array([i + 1] * self.n_query),
          ),
        )


def create_mlc_dataloader(
  n_primitives: int,
  n_rules: int,
  vocabulary: Literal['MLC', 'LETTERS', 'NUMBERS'],
  max_rhs_length: int,
  max_example_length: int,
  p_lhs_onearg: float,
  n_languages: int,
  n_translation_per_language: int,
  n_recomposition_per_language: int,
  n_shuffle_per_language: int,
  n_support: int,
  n_query: int,
  *,
  seed: int,
) -> tuple[MLCDataloader, str, type[pydantic.BaseModel]]:
  dataloader = MLCDataloader(
    n_primitives=n_primitives,
    n_rules=n_rules,
    vocabulary=vocabulary,
    max_rhs_length=max_rhs_length,
    max_example_length=max_example_length,
    p_lhs_onearg=p_lhs_onearg,
    n_languages=n_languages,
    n_translation_per_language=n_translation_per_language,
    n_recomposition_per_language=n_recomposition_per_language,
    n_shuffle_per_language=n_shuffle_per_language,
    n_support=n_support,
    n_query=n_query,
    seed=seed,
  )
  system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n_support=n_support, n_query=n_query)
  schema = pydantic.create_model(
    'Answer', **{str(i + 1): (str, ...) for i in range(n_query)}
  )

  return (dataloader, system_prompt, schema)


if __name__ == '__main__':
  input_symbols_list_default = (
    'dax',
    'lug',
    'wif',
    'zup',
    'fep',
    'blicket',
    'kiki',
    'tufa',
    'gazzer',
  )
  output_symbols_list_default = ('RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK')

  # Initialize the generator
  meta_grammar = MetaGrammar(
    n_primitives=4,
    n_rules=3,
    input_vocab=input_symbols_list_default,
    output_vocab=output_symbols_list_default,
    max_rhs_length=3,
    max_example_length=16,
    p_lhs_onearg=0.5,
    seed=0,
  )

  translator = Translator(seed=0)

  print('Generating unique artificial languages and their episodes...\n')

  for i in range(3):
    # Sample a unique language
    language = meta_grammar.sample()

    # Generate an episode
    episode = language.generate_episode(n_support=14, n_query=10, max_tries=10000)

    print(f'--- Language {i + 1} ---')
    print('Primitives:')
    for p_in, p_out in language.primitives:
      print(f'  {p_in} -> {p_out}')

    print('\nSignature:\n' + language.hash_signature)

    print('\nSupport Examples:')
    for idx, _ in enumerate(episode.support_in):
      print(f'  IN: {episode.support_in[idx]}  -->  OUT: {episode.support_out[idx]}')

    print('Query Examples:')
    for idx, _ in enumerate(episode.query_in):
      print(f'  IN: {episode.query_in[idx]}  -->  OUT: {episode.query_out[idx]}')

    print('\nTranslated episode:')
    translated_episode = translator.random_translation(
      episode,
      input_symbols_old=input_symbols_list_default,
      input_symbols_new=input_symbols_list_default,
      output_symbols_old=output_symbols_list_default,
      output_symbols_new=output_symbols_list_default,
    )
    print('Support Examples:')
    for idx, _ in enumerate(translated_episode.support_in):
      print(
        f'  IN: {translated_episode.support_in[idx]}  -->  '
        f'OUT: {translated_episode.support_out[idx]}'
      )
    print('Query Examples:')
    for idx, _ in enumerate(translated_episode.query_in):
      print(
        f'  IN: {translated_episode.query_in[idx]}  -->  '
        f'OUT: {translated_episode.query_out[idx]}'
      )

    print('\n' + '=' * 40 + '\n')
