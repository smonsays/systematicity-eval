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

import math

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt
from absl.testing import absltest
from absl.testing import parameterized

from synthsys.data import raven
from synthsys.model.api import APIModel


class RavenTestCase(parameterized.TestCase):
  rng = jax.random.key(0)

  def test_sample_instance(self) -> None:
    with jax.disable_jit(False):
      generator = raven.SymbolicRavenGenerator(
        n_features := 5,
        feature_maxval := 8,
        grid_size := 3,
        frac_ood=0.25,
        seed=0,
      )

      latents = jnp.arange(generator.n_rules)
      rngs = jax.random.split(self.rng, len(latents))
      instances = generator.sample_instance(rngs, latents)

      assert instances.shape == (generator.n_rules, grid_size, grid_size)
      assert jnp.all(instances < feature_maxval)
      assert jnp.all(instances >= 0)

  @parameterized.parameters(dict(permute=True), dict(permute=False))
  def test_sample(self, permute: bool, batch_size: int = 7) -> None:
    with jax.disable_jit(False):
      generator = raven.SymbolicRavenGenerator(
        n_features := 5,
        feature_maxval := 5,
        grid_size := 3,
        frac_ood=0.25,
        seed=0,
      )
      tasks = generator.sample(self.rng, batch_size, latent_dist='train', permute=permute)
      assert tasks.instances.shape == (batch_size, grid_size, grid_size, n_features)
      assert jnp.max(tasks.instances) <= 5
      assert tasks.latents.shape == (batch_size, n_features)
      assert tasks.permutation_ids.shape == (batch_size,)

  def test_format_as_prompt(self) -> None:
    generator = raven.SymbolicRavenGenerator(
      n_features := 5,
      feature_maxval=20,
      grid_size=3,
      frac_ood=0.0,
      seed=0,
    )
    task = generator.sample(self.rng, n_tasks=1, latent_dist='train', permute=False)
    instance = task.instances.squeeze()
    prompt_masked = raven.format_as_prompt(instance, n_features, mask_answer=True)
    prompt_unmasked = raven.format_as_prompt(instance, n_features, mask_answer=False)

    print(prompt_masked)
    print(prompt_unmasked)

  def test_raven_dataloader(self) -> None:
    dataloader = raven.RavenDataloader(
      n_sets := 100,
      n_permutations_per_set := 5,
      n_features := 2,
      feature_maxval := 10,
      seed=0,
    )
    assert len(list(dataloader)) == n_sets * n_permutations_per_set

    for batch in dataloader:
      # print(batch)
      assert jnp.min(batch.target) >= 0
      assert jnp.max(batch.target) < feature_maxval
      chex.assert_shape(batch.target, (n_features,))
      chex.assert_shape(batch.rule, (n_features,))
      chex.assert_shape(batch.info['permutation_id'], (n_features,))
      chex.assert_shape(batch.info['set_id'], (n_features,))

  def test_create_raven_dataloader(self) -> None:
    dataloader, prompt, schema = raven.create_raven_dataloader(
      n_sets := 7,
      n_permutations_per_set := 2,
      n_features := 3,
      feature_maxval := 5,
      seed=0,
    )
    print(prompt)

  @absltest.skip('Skipping long-running ambiguity test')
  def test_is_ambiguous(
    self,
    n_features: int = 2,
    feature_maxval: int = 10,
    permute: bool = True,
    batch_size: int = 2,
    n_batches: int = 64,
    seed: int = 0,
  ) -> None:
    with jax.disable_jit(False):
      rng = jax.random.key(seed)
      generator = raven.SymbolicRavenGenerator(
        n_features=n_features,
        feature_maxval=feature_maxval,
        grid_size=3,
        frac_ood=0.0,
        seed=seed,
      )

      def scan_ambiguous(
        _: None, r: jt.PRNGKeyArray
      ) -> tuple[None, jt.Int[jt.Array, ' nbatches']]:
        task = generator.sample(r, batch_size, latent_dist='train', permute=permute)
        n_ambiguous = generator.check_is_ambiguous(task.instances)
        return None, n_ambiguous

      _, n_ambiguous = jax.lax.scan(
        scan_ambiguous, None, jax.random.split(rng, n_batches)
      )

      ambig_mean = jnp.mean(n_ambiguous > 0)
      ambig_sem = jnp.std(n_ambiguous > 0) / math.sqrt(n_ambiguous.size)
      print('\n Fraction of ambiguous instances: {}±{}'.format(ambig_mean, ambig_sem))

  @absltest.skip('Skipping long-running sample response test')
  @parameterized.parameters(
    dict(
      n_sets=1,
      n_permutations_per_set=1,
      n_features=3,
      feature_maxval=5,
      seed=0,
      model_name='gemini:gemini-3.1-flash-lite-preview',
      reasoning_effort='low',
    )
  )
  def test_sample_responses(
    self,
    n_sets: int,
    n_permutations_per_set: int,
    n_features: int,
    feature_maxval: int,
    seed: int,
    model_name: str,
    reasoning_effort: str,
  ) -> None:
    model = APIModel(
      model_name=model_name,
      reasoning_effort=reasoning_effort,
      temperature=0,
    )

    dataloader, system_prompt, _ = raven.create_raven_dataloader(
      n_sets=n_sets,
      n_permutations_per_set=n_permutations_per_set,
      n_features=n_features,
      feature_maxval=feature_maxval,
      seed=seed,
    )
    print(system_prompt)
    for batch in dataloader:
      response = model(batch.prompt, system_prompt)
      print('\n===PROMPT===')
      print(batch.prompt)
      print('\n===RESPONSE===')
      print(response)
      print('\n===TARGET===')
      print(batch.target)


if __name__ == '__main__':
  absltest.main()
