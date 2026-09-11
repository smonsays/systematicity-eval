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

"""Logging interfaces."""

import abc
import dataclasses
import json
import os
from datetime import datetime

import jaxtyping as jt
import pandas as pd
import wandb
from absl import logging

from synthsys.config_classes import Config


def run_name(config: Config) -> str:
  return (
    config.dataset.name
    + '_'
    + config.model.name.replace('/', '-').replace(':', '-')
    + '_'
    + datetime.now().strftime('%Y%m%d_%H%M%S')
  )


class Logger(abc.ABC):
  @abc.abstractmethod
  def log(self, step: int, metrics: dict[str, jt.ArrayLike]) -> None:
    pass

  @abc.abstractmethod
  def store(self, config: Config, results: pd.DataFrame) -> None:
    """Store results and corresponding config."""
    pass


@dataclasses.dataclass
class LocalLogger(Logger):
  path: str

  def log(self, step: int, metrics: dict[str, jt.ArrayLike]) -> None:
    """Log to stdout"""
    metrics = {k: v for k, v in metrics.items() if isinstance(v, (jt.ArrayLike))}
    metrics_str = ' \t '.join('{}: {:.4f}'.format(str(k), v) for k, v in metrics.items())
    logging.info('step: {} \t'.format(step) + metrics_str)

  def store(self, config: Config, results: pd.DataFrame) -> None:
    """Stores config and results on disk as json and parquet respectively."""
    filename = run_name(config)
    logging.info(f'Storing results in {filename}')

    results.to_parquet(os.path.join(self.path, filename + '_results.parquet'))

    config_dict = dataclasses.asdict(config)
    with open(os.path.join(self.path, filename + '_config.json'), 'w') as f:
      json.dump(config_dict, f, indent=2)


class WandbLogger(Logger):
  def __init__(self, config: Config) -> None:
    super().__init__()
    wandb.init(
      config=dataclasses.asdict(config),
      save_code=False,
    )

  def log(self, step: int, metrics: dict[str, jt.ArrayLike]) -> None:
    wandb.log({**metrics, 'step': step}, step=step)

  def store(self, config: Config, results: pd.DataFrame) -> None:
    """Stores config and results as artifacts in wandb."""
    artifact = wandb.Artifact(
      name=run_name((config)),
      type='result',
    )
    artifact.add(wandb.Table(dataframe=results), 'results')
    wandb.log_artifact(artifact)
