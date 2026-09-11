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

"""Data generation for raven rule induction tasks."""

import hashlib
import logging
from functools import partial
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
from typing import Callable
from typing import Iterator

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pydantic
from einops import rearrange
from flax import struct

from synthsys.data.base import Batch

SYSTEM_PROMPT_TEMPLATE = """
Your task is to predict integers in structured sequences of integers.
Specifically, you will be presented with tasks that consist of three rows with three columns each.
Each panel in this three by three matrix contains {n_features} feature(s) encoded as integer(s).
Within a row the {n_features} feature(s) of the last column can be predicted by applying the correct rules to the first two columns.
Within a task, all rows follow the same underlying rules applied to different inputs.

Each row is delimited by ||, each column is delimited by | and the features within a panel are separated by spaces. The last {n_features} feature(s) in the last panel are masked with {n_features} question marks, '?'.

1. Try to identify which integers in a given task are governed by the same underlying rules.
2. Try to identify the {n_features} rule(s) that govern a task.
3. Consider the following rules/patterns:
   - Constant, e.g. the same integer is repeated
   - Progression, e.g. integers are systematically incremented or decremented by steps of +1, +2, -1, or -2
   - Modular addition, e.g. two integers prior in the sequence are summed to produce the next integer
   - Modular subtraction, e.g. two integers prior in the sequence are subtracted to produce the next integer
   - Maximum/Minimum, e.g. the maximum/minimum of two integers prior in the sequence makes up the next integer
"""  # noqa: E501


def _digit_to_str(n: int) -> str:
  mapping = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
  }
  return mapping[n]


@struct.dataclass(frozen=True)
class RavenTask:
  instances: jt.Int[jt.Array, '... row column feature']
  latents: jt.Int[jt.Array, '... feature']
  permutation_ids: jt.Int[jt.Array, '...']


class SymbolicRavenGenerator:
  def __init__(
    self,
    n_features: int,
    feature_maxval: int,
    grid_size: int,
    frac_ood: float,
    seed: int,
  ) -> None:
    self.n_features = n_features
    self.feature_maxval = feature_maxval
    self.grid_size = grid_size
    self.frac_ood = frac_ood
    self.fixed_rng = jax.random.key(seed)

    # Generate in- and out-dist latents
    self.latents_all = jnp.array(
      list(combinations_with_replacement(range(self.n_rules), n_features))
    )
    self.n_latents = len(self.latents_all)
    self.n_ood = int(self.n_latents * self.frac_ood)

    latents_idx_all = jax.random.permutation(self.fixed_rng, jnp.arange(self.n_latents))
    self.latents_idx_in_dist = latents_idx_all[self.n_ood :]
    self.latents_idx_out_dist = latents_idx_all[: self.n_ood]

    logging.info('SymbolicRavenGenerator initialized with:')
    logging.info('{} in-dist latents'.format(len(self.latents_idx_in_dist)))
    logging.info('{} out-dist latents'.format(len(self.latents_idx_out_dist)))

    if len(jnp.unique(self.latents_all[self.latents_idx_in_dist])) < self.n_rules:
      logging.warning('Not all rules contained in in-dist set.')

    if len(jnp.unique(self.latents_all[self.latents_idx_out_dist])) < self.n_rules:
      logging.warning('Not all rules contained in ood set.')

    assert len(self.latents_idx_in_dist) > 0, 'In-dist set is empty'
    if self.frac_ood > 0:
      assert len(self.latents_idx_out_dist) > 0, 'OOD set is empty'

  @property
  def n_rules(self) -> int:
    return len(self.rules)

  @property
  def rules(self) -> dict[str, Callable[[jt.PRNGKeyArray], jt.Int[jt.Array, 'N M']]]:
    """
    Each rule is a function that returns an array of shape (grid_size, grid_size).
    """
    n_examples, seq_len = self.grid_size, self.grid_size
    maxval = self.feature_maxval

    def constant(rng: jt.PRNGKeyArray) -> jt.Int[jt.Array, 'N M']:
      const = jax.random.randint(rng, shape=(n_examples, 1), minval=0, maxval=maxval)
      return jnp.broadcast_to(const, shape=(n_examples, seq_len))

    def progression(rng: jt.PRNGKeyArray, *, inc: int) -> jt.Int[jt.Array, 'N M']:
      start = jax.random.randint(rng, shape=(n_examples, 1), minval=0, maxval=maxval)
      return (start + inc * jnp.arange(0, seq_len)[jnp.newaxis]) % maxval

    def arithmetic(rng: jt.PRNGKeyArray, *, subtract: bool) -> jt.Int[jt.Array, 'N M']:
      xs = jax.random.randint(
        rng, shape=(n_examples, seq_len - 1), minval=0, maxval=maxval
      )
      if subtract:
        xs = xs.at[:, 1:].set(xs[:, 1:] * (-1))

      res = jnp.sum(xs, keepdims=True, axis=1) % maxval
      return jnp.concatenate((jnp.abs(xs), res), axis=-1)

    def distribute_three(rng: jt.PRNGKeyArray) -> jt.Int[jt.Array, 'N M']:
      rng_choice, rng_perm = jax.random.split(rng)
      symbols = jax.random.choice(rng_choice, maxval, shape=(seq_len,), replace=False)
      symbols = jnp.broadcast_to(symbols, shape=(n_examples, seq_len))
      return jax.random.permutation(rng_perm, symbols, axis=1, independent=True)

    def minimum(rng: jt.PRNGKeyArray) -> jt.Int[jt.Array, 'N M']:
      xs = jax.random.randint(
        rng, shape=(n_examples, seq_len - 1), minval=0, maxval=maxval
      )
      res = jnp.min(xs, keepdims=True, axis=1)
      return jnp.concatenate((xs, res), axis=-1)

    def maximum(rng: jt.PRNGKeyArray) -> jt.Int[jt.Array, 'N M']:
      xs = jax.random.randint(
        rng, shape=(n_examples, seq_len - 1), minval=0, maxval=maxval
      )
      res = jnp.max(xs, keepdims=True, axis=1)
      return jnp.concatenate((xs, res), axis=-1)

    return {
      'constant': constant,
      'progression_plus_one': partial(progression, inc=1),
      'progression_plus_two': partial(progression, inc=2),
      'progression_minus_one': partial(progression, inc=-1),
      'progression_minus_two': partial(progression, inc=-2),
      'addition': partial(arithmetic, subtract=False),
      'subtraction': partial(arithmetic, subtract=True),
      'distribute_three': distribute_three,
      'minimum': minimum,
      'maximum': maximum,
    }

  @partial(jnp.vectorize, excluded=(0,), signature='(),()->(n,n)')
  def sample_instance(self, rng: jt.PRNGKeyArray, latent: jax.Array) -> jax.Array:
    """
    Given a rule specified by latent, sample a valid corresponding instance.
    """
    return jax.lax.switch(latent, self.rules.values(), rng)

  def sample_latents(
    self, rng: jt.PRNGKeyArray, n_tasks: int, latent_dist: str
  ) -> jt.Int[jt.Array, 'task feature']:
    match latent_dist:
      case 'train' | 'test':
        latents_idx_all = self.latents_idx_in_dist
      case 'ood':
        latents_idx_all = self.latents_idx_out_dist
      case 'ind+ood':
        latents_idx_all = jnp.concatenate(
          (self.latents_idx_in_dist, self.latents_idx_out_dist), axis=0
        )
      case _:
        raise ValueError(f'Invalid latent_dist: {latent_dist}')

    latents_idx = jax.random.choice(rng, latents_idx_all, shape=(n_tasks,), replace=True)

    return self.latents_all[latents_idx]

  def sample_permutation_ids(
    self, rng: jt.PRNGKeyArray, n_tasks: int
  ) -> jt.Int[jt.Array, ' task']:
    max_int = jnp.iinfo(jnp.int32).max
    return jax.random.randint(rng, shape=(n_tasks,), minval=0, maxval=max_int)

  def permute_instances(
    self,
    permutation_ids: jt.Int[jt.Array, ' task'],
    instances: jt.Int[jt.Array, 'task row column feature'],
  ) -> jt.Int[jt.Array, 'task row column feature']:
    """Permute column features using a consistent permutation across rows."""

    @partial(jnp.vectorize, signature=('(),(r,f)->(r,f)'))
    def permute_features_for_all_rows(
      rng: jt.PRNGKeyArray, x: jt.Int[jt.Array, 'row feature']
    ) -> jax.Array:
      """Apply a consistent permutation of the features across rows given one column."""
      return x[:, jax.random.permutation(rng, jnp.arange(self.n_features))]

    @partial(jnp.vectorize, signature=('()->(c)'))
    def id_to_rngs(permutation_id: jt.Int[jt.Array, '']) -> jt.Key[jt.Array, ' column']:
      """Map permutation id to one rng per column."""
      return jax.random.split(jax.random.key(permutation_id), self.grid_size)

    rngs_perm = id_to_rngs(permutation_ids)  # shape=(n_tasks, self.grid_size)

    # The permutation needs to be consistent across n_examples.
    # We swap axes so we can vmap over seq_len and swap back afterwards.
    instances = jnp.swapaxes(instances, 1, 2)
    instances = permute_features_for_all_rows(rngs_perm, instances)
    instances = jnp.swapaxes(instances, 1, 2)

    return instances

  @partial(jax.jit, static_argnames=('self', 'n_tasks', 'latent_dist', 'permute'))
  def sample(
    self, rng: jt.PRNGKeyArray, n_tasks: int, latent_dist: str, permute: bool
  ) -> RavenTask:
    rng_tasks, rng_samples, rng_perm = jax.random.split(rng, 3)
    latents = self.sample_latents(rng_tasks, n_tasks, latent_dist)

    # Sample instances for each task and for each feature dimension
    rngs_samples = jax.random.split(rng_samples, latents.size).reshape(latents.shape)
    instances = self.sample_instance(rngs_samples, latents)
    instances = rearrange(instances, 'tasks feat rows cols -> tasks rows cols feat')

    # Sample permutation
    if permute:
      permutation_ids = self.sample_permutation_ids(rng_perm, n_tasks)
      instances = self.permute_instances(permutation_ids, instances)
    else:
      permutation_ids = -jnp.ones((n_tasks,))

    return RavenTask(instances, latents, permutation_ids)

  def check_is_ambiguous(self, instances: jax.Array) -> jax.Array:
    """
    Takes a batch of sraven problem instances and checks whether they have ambiguous
    answers, i.e. whether there are multiple sets of rules that fit the query but return
    different answers.
    """

    def is_constant(x: jax.Array):
      split_index = (self.grid_size - 1) * self.grid_size
      first_rows, last_row = x[:split_index], x[split_index:]

      # Check first two rows
      first_rows = rearrange(
        first_rows, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size
      )
      first_rows_const = jnp.all(jax.vmap(lambda x: jnp.all(x == x[0]))(first_rows))
      last_row_const = jnp.all(last_row == last_row[0])

      return jnp.logical_and(first_rows_const, last_row_const), last_row[0]

    def is_progression(x: jax.Array):
      diff_mod = jnp.diff(x) % self.feature_maxval
      diff_mod_masked = diff_mod[(np.arange(self.grid_size**2 - 2) + 1) % 3 != 0]

      step_size = diff_mod_masked[0]
      same_step_size = jnp.all(diff_mod_masked == step_size)
      step_size_in_range = (
        (step_size == 1)
        | (step_size == 2)
        | (step_size == -1 % self.feature_maxval)
        | (step_size == -2 % self.feature_maxval)
      )
      next_step = (x[-1] + step_size) % self.feature_maxval
      return jnp.logical_and(same_step_size, step_size_in_range), next_step

    def is_max(
      x: jt.Int[jt.Array, ' query'],
    ) -> tuple[jt.Bool[jt.Array, ''], jt.Int[jt.Array, '']]:
      @jax.vmap
      def is_max_(y: jax.Array) -> jax.Array:
        return (jnp.max(y[:2])) == y[2]

      split_index = (self.grid_size - 1) * self.grid_size
      first_rows, last_row = x[:split_index], x[split_index:]
      first_rows = rearrange(
        first_rows, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size
      )
      return jnp.all(is_max_(first_rows)), jnp.max(last_row)

    def is_min(
      x: jt.Int[jt.Array, ' query'],
    ) -> tuple[jt.Bool[jt.Array, ''], jt.Int[jt.Array, '']]:
      @jax.vmap
      def is_min_(y: jax.Array) -> jax.Array:
        return (jnp.min(y[:2])) == y[2]

      split_index = (self.grid_size - 1) * self.grid_size
      first_rows, last_row = x[:split_index], x[split_index:]
      first_rows = rearrange(
        first_rows, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size
      )
      return jnp.all(is_min_(first_rows)), jnp.min(last_row)

    def is_sum(x: jax.Array):
      @jax.vmap
      def is_sum_(y: jax.Array):
        return (jnp.sum(y[:2]) % self.feature_maxval) == y[2]

      y = x[: (self.grid_size - 1) * self.grid_size]
      y = rearrange(y, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size)
      return jnp.all(is_sum_(y)), (x[-2] + x[-1]) % self.feature_maxval

    def is_difference(x: jax.Array):
      @jax.vmap
      def is_difference_(y: jax.Array):
        y = y.at[1:-1].set(y[1:-1] * (-1))
        return (jnp.sum(y[:2]) % self.feature_maxval) == y[2]

      y = x[: (self.grid_size - 1) * self.grid_size]
      y = rearrange(y, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size)
      return jnp.all(is_difference_(y)), (x[-2] - x[-1]) % self.feature_maxval

    def is_distribute_three(x: jax.Array):
      split_index = (self.grid_size - 1) * self.grid_size
      first_rows, last_row = x[:split_index], x[split_index:]

      # Check first two rows
      first_rows = rearrange(
        first_rows, '(n m) -> n m', n=self.grid_size - 1, m=self.grid_size
      )
      first_rows_sorted = jnp.sort(first_rows, axis=1)
      first_rows_true = jnp.all(first_rows_sorted[0] == first_rows_sorted)

      # Check last row
      possible_vals_sorted = first_rows_sorted[0]
      last_row_matches = last_row[:, jnp.newaxis] == possible_vals_sorted[jnp.newaxis, :]
      last_row_true = jnp.all(jnp.any(last_row_matches, axis=1))
      answer = possible_vals_sorted[jnp.argmin(jnp.sum(last_row_matches, axis=0))]

      return jnp.logical_and(first_rows_true, last_row_true), answer

    @partial(jnp.vectorize, signature='(n)->(r),(r)')
    def check_all_rules(x: jax.Array):
      rule_applies, rule_answer = zip(
        is_constant(x),
        is_progression(x),
        is_sum(x),
        is_difference(x),
        is_distribute_three(x),
        is_max(x),
        is_min(x),
        strict=True,
      )
      return jnp.array(rule_applies), jnp.array(rule_answer)

    def check_all_rules_given_perm(count: jax.Array, perm: jax.Array):
      instances_perm = jnp.vectorize(lambda x, p: x[p], signature='(m),(m)->(m)')(
        instances, perm
      )
      query = rearrange(instances_perm, 'b n s m -> b m (n s)')[:, :, :-1]
      answer = rearrange(instances_perm, 'b n s m -> b m (n s)')[:, :, -1]

      rule_applies, rule_answer = check_all_rules(query)
      answer_differs = answer[:, :, jnp.newaxis] != rule_answer
      any_rule_fits_a_feature = jnp.logical_and(rule_applies, answer_differs).any(axis=2)
      all_features_fit_by_rule = jnp.all(any_rule_fits_a_feature, axis=1)

      count += all_features_fit_by_rule

      return count, None

    # scanning over all possible permutations
    # NOTE: This quickly becomes intractable for large `n_features`
    perms = jnp.array(
      list(product(permutations(range(self.n_features)), repeat=self.grid_size))
    )
    count, _ = jax.lax.scan(check_all_rules_given_perm, jnp.zeros(len(instances)), perms)
    return count


class RavenDataloader:
  def __init__(
    self,
    n_sets: int,
    n_permutations_per_set: int,
    n_features: int,
    feature_maxval: int,
    seed: int,
  ) -> None:
    if n_permutations_per_set > 0 and n_features == 1:
      raise ValueError('Permutations can only be created for n_features > 1')
    self.generator = SymbolicRavenGenerator(
      n_features, feature_maxval, grid_size=3, frac_ood=0.0, seed=seed
    )
    self.n_sets = n_sets
    self.n_permutations_per_set = n_permutations_per_set
    self.seed = seed

  def __iter__(self) -> Iterator[Batch]:
    n_features = self.generator.n_features
    rng = jax.random.key(self.seed)

    sets_yielded = 0
    while sets_yielded < self.n_sets:
      rng, rng_set = jax.random.split(rng)
      task = self.generator.sample(rng_set, n_tasks=1, latent_dist='train', permute=False)

      valid_perms = []
      for _ in range(100 * self.n_permutations_per_set):
        if len(valid_perms) == self.n_permutations_per_set:
          break

        rng_set, rng_perm = jax.random.split(rng_set)
        permutation_id = self.generator.sample_permutation_ids(rng_perm, 1)
        instance_perm = self.generator.permute_instances(permutation_id, task.instances)

        if not jnp.all(instance_perm == task.instances) and not jnp.all(
          instance_perm == task.instances[..., ::-1]
        ):
          valid_perms.append((permutation_id, instance_perm))

      if len(valid_perms) < self.n_permutations_per_set:
        continue

      set_id = hashlib.md5(
        np.array(task.latents).tobytes() + np.array(task.instances).tobytes()
      ).hexdigest()

      for idx, (permutation_id, instance_perm) in enumerate(valid_perms):
        instance_perm = rearrange(instance_perm, '1 r c f -> r c f')

        yield Batch(
          prompt=format_as_prompt(instance_perm, n_features, mask_answer=True),
          target=instance_perm[-1, -1, :],
          rule=rearrange(task.latents, '1 f -> f'),
          info=dict(
            set_id=np.array([set_id] * n_features),
            variation_type=np.array(
              [('canonical' if idx == 0 else 'permutation')] * n_features
            ),
            variation_idx=np.array([idx] * n_features),
            permutation_id=np.array([permutation_id.squeeze()] * n_features),
          ),
        )

      sets_yielded += 1


def format_as_prompt(
  instance: jt.Integer[jt.Array, 'row col feature'],
  n_features: int,
  mask_answer: bool,
) -> str:
  assert len(instance.shape) == 3
  assert np.issubdtype(instance.dtype, np.integer)

  formatted_rows = []
  for row in instance:
    formatted_cols = []
    for col in row:
      formatted_cols.append(str(col)[1:-1])  # removes brackets
    formatted_rows.append(' | '.join(formatted_cols))

  prompt = ' || '.join(formatted_rows)

  if mask_answer:
    query_str_len = len(str(instance[-1, -1])[1:-1])
    prompt = prompt[:-query_str_len] + ' '.join(['?' for _ in range(n_features)])

  return prompt


def create_raven_dataloader(
  n_sets: int,
  n_permutations_per_set: int,
  n_features: int,
  feature_maxval: int,
  seed: int,
) -> tuple[RavenDataloader, str, type[pydantic.BaseModel]]:
  dataloader = RavenDataloader(
    n_sets=n_sets,
    n_permutations_per_set=n_permutations_per_set,
    n_features=n_features,
    feature_maxval=feature_maxval,
    seed=seed,
  )

  system_prompt = SYSTEM_PROMPT_TEMPLATE.format(n_features=_digit_to_str(n_features))
  schema = pydantic.create_model('Answer', **{str(i + 1): int for i in range(n_features)})

  return (dataloader, system_prompt, schema)
