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

"""Entry point for running systematicity evaluation experiments."""

import asyncio
import dataclasses
from collections import defaultdict

import einops
import jax
import jaxtyping as jt
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from ml_collections import config_flags

from synthsys import config_classes
from synthsys.data.boolean import create_boolean_dataloader
from synthsys.data.listint import create_listint_dataloader
from synthsys.data.mlc import create_mlc_dataloader
from synthsys.data.raven import create_raven_dataloader
from synthsys.logger import LocalLogger
from synthsys.logger import WandbLogger
from synthsys.metrics import compute_systematicity_metrics
from synthsys.model.api import AsyncAPIModel

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
  name='config',
  default='configs/mlc.py',
  help_string='Experiment configuration.',
)


@dataclasses.dataclass
class Results:
  answers: jt.Int[np.ndarray, 'task attempt feature']
  targets: jt.Int[np.ndarray, 'task feature']
  rules: jt.Shaped[np.ndarray, 'task feature']
  prompts: jt.Shaped[np.ndarray, ' task']
  explanations: jt.Shaped[np.ndarray, 'task attempt']
  infos: dict[str, jt.Shaped[np.ndarray, 'task feature']]


async def run_evaluation(config: config_classes.Config) -> Results:
  model = AsyncAPIModel(
    config.model.name,
    provider=config.model.provider,
    max_concurrent=config.model.max_concurrent,
    reasoning_effort=config.model.reasoning_effort,  # ty:ignore[invalid-argument-type]
    temperature=config.model.temperature,
  )

  match config.dataset:
    case config_classes.BooleanDatasetConfig():
      dataloader, system_prompt, response_schema = create_boolean_dataloader(
        n_features=config.dataset.n_features,
        feature_maxval=config.dataset.feature_maxval,
        n_query=config.dataset.n_query,
        min_parse_tree_size=config.dataset.min_parse_tree_size,
        max_parse_tree_size=config.dataset.max_parse_tree_size,
        n_rules_per_size=config.dataset.n_rules_per_size,
        n_translations_per_rule=config.dataset.n_translations_per_rule,
        n_shuffle_per_rule=config.dataset.n_shuffle_per_rule,
        seed=config.seed,
      )
    case config_classes.MLCDatasetConfig():
      dataloader, system_prompt, response_schema = create_mlc_dataloader(
        n_primitives=config.dataset.n_primitives,
        n_rules=config.dataset.n_rules,
        vocabulary=config.dataset.vocabulary,  # type: ignore[arg-type]
        max_rhs_length=config.dataset.max_rhs_length,
        max_example_length=config.dataset.max_example_length,
        p_lhs_onearg=config.dataset.p_lhs_onearg,
        n_languages=config.dataset.n_languages,
        n_translation_per_language=config.dataset.n_translation_per_language,
        n_recomposition_per_language=config.dataset.n_recomposition_per_language,
        n_shuffle_per_language=config.dataset.n_shuffle_per_language,
        n_support=config.dataset.n_support,
        n_query=config.dataset.n_query,
        seed=config.seed,
      )
    case config_classes.RavenDatasetConfig():
      dataloader, system_prompt, response_schema = create_raven_dataloader(
        n_sets=config.dataset.n_sets,
        n_permutations_per_set=config.dataset.n_permutations_per_set,
        n_features=config.dataset.n_features,
        feature_maxval=config.dataset.feature_maxval,
        seed=config.seed,
      )
    case config_classes.ListintDatasetConfig():
      dataloader, system_prompt, response_schema = create_listint_dataloader(
        n_support=config.dataset.n_support,
        n_query=config.dataset.n_query,
        min_parse_tree_size=config.dataset.min_parse_tree_size,
        max_parse_tree_size=config.dataset.max_parse_tree_size,
        n_rules_per_size=config.dataset.n_rules_per_size,
        n_translation_per_rule=config.dataset.n_translation_per_rule,
        n_shuffle_per_rule=config.dataset.n_shuffle_per_rule,
        list_maxlength=config.dataset.list_maxlength,
        feature_maxval=config.dataset.feature_maxval,
        easy_instructions=config.dataset.easy_instructions,
        seed=config.seed,
      )
    case _:
      raise ValueError(f'Undefined dataset config: {config.dataset}')

  tasks, prompts, targets, rules = [], [], [], []
  infos = defaultdict(list)

  for batch in dataloader:
    for _ in range(config.model.n_attempts):
      task = model.respond_maybe_with_explanation(
        prompt=batch.prompt,
        response_schema=response_schema,
        system_prompt=system_prompt,
        explain=config.model.explain,
      )
      tasks.append(task)

    # Store metadata once per unique task (not per attempt)
    prompts.append(batch.prompt)
    targets.append(batch.target)
    rules.append(batch.rule)

    for key, value in batch.info.items():
      infos[key].append(value)

  responses = await asyncio.gather(*tasks)

  # Collect `Results` in shaped numpy arrays
  n_tasks, n_attempts = len(prompts), config.model.n_attempts
  answers = np.array([np.array([val for _, val in r.response]) for r in responses])
  explanations = np.array([r.explanation for r in responses])

  return Results(
    answers=einops.rearrange(answers, '(t a) f -> t a f', t=n_tasks, a=n_attempts),
    targets=np.stack(targets),
    rules=np.stack(rules),
    prompts=np.stack(jax.tree.map(np.array, prompts)),
    explanations=einops.rearrange(explanations, '(t a) -> t a', t=n_tasks, a=n_attempts),
    infos=dict((key, np.stack(values)) for key, values in infos.items()),
  )


def main(argv: list[str]) -> None:
  del argv
  # logging.info('Running on {}'.format(jax.default_backend()))
  config: config_classes.Config = flags.FLAGS.config

  match config.logger:
    case 'local':
      logger = LocalLogger(path='results')
    case 'wandb':
      logger = WandbLogger(config=config)
    case None:
      logger = None
    case _:
      raise ValueError(f'Undefined storage type: {config.logger}')

  results = asyncio.run(run_evaluation(config))

  # Store results in dataframe to be persisted to disk
  n_tasks, n_attempts, n_features = results.answers.shape

  df = pd.DataFrame(
    dict(
      answer=results.answers.flatten(),
      target=einops.repeat(results.targets, 't f -> (t a f)', a=n_attempts).flatten(),
      rule=einops.repeat(results.rules, 't f -> (t a f)', a=n_attempts).flatten(),
      prompt=einops.repeat(results.prompts, 't -> (t a f)', a=n_attempts, f=n_features),
      explanation=einops.repeat(results.explanations, 't a -> (t a f)', f=n_features),
      task_idx=einops.repeat(
        np.arange(n_tasks), 't -> (t a f)', a=n_attempts, f=n_features
      ),
      attempt_idx=einops.repeat(
        np.arange(n_attempts), 'a -> (t a f)', t=n_tasks, f=n_features
      ),
      feature_idx=einops.repeat(
        np.arange(n_features), 'f -> (t a f)', t=n_tasks, a=n_attempts
      ),
      **{
        key: einops.repeat(values, 't f -> (t a f)', a=n_attempts).flatten()
        for key, values in results.infos.items()
      },
    )
  )
  df['correct'] = df['answer'] == df['target']
  df['correct_all'] = df.groupby(['task_idx', 'attempt_idx'])['correct'].transform('all')

  if 'map_target' in df.columns:
    df['correct_map'] = df['answer'] == df['map_target']
    df['correct_map_all'] = df.groupby(['task_idx', 'attempt_idx'])[
      'correct_map'
    ].transform('all')

  # Add all config values as columns to the results dataframe
  config_flat_dict = pd.json_normalize((dataclasses.asdict(config))).iloc[0].to_dict()
  df = df.assign(**config_flat_dict)
  df['config_path'] = config_flags.get_config_filename(FLAGS['config'])

  # Compute metrics and print a summary
  tasks_passed = df.groupby(['task_idx', 'attempt_idx'])['correct_all'].first().mean()

  print('\n' + '=' * 65)
  print(f'EVALUATION SUMMARY: {config.dataset.name} | Model: {config.model.name}')
  print('=' * 65)
  print(f'Raw task pass rate: {tasks_passed:.2%}\n')

  variation_types = [v for v in df['variation_type'].unique() if v != 'canonical']

  if variation_types:
    summary_metrics = compute_systematicity_metrics(df)
    print('Systematicity metrics:')
    print(summary_metrics.to_string(index=False, float_format=lambda x: f'{x:.2%}'))
  else:
    print('No systematic variations were included in this evaluation.')

  print('=' * 65 + '\n')

  if logger:
    metrics = {'acc/raw_attempt': tasks_passed}
    if variation_types:
      overall_row = summary_metrics[summary_metrics['Variation'] == 'Overall (Macro)']
      metrics.update(
        {
          'systematicity/pass_any': float(overall_row['pass_any'].values[0]),
          'systematicity/pass_frac': float(overall_row['pass_frac'].values[0]),
          'systematicity/pass_all': float(overall_row['pass_all'].values[0]),
        }
      )
    logger.log(step=0, metrics=metrics)
    logger.store(config, df)


if __name__ == '__main__':
  app.run(main)
