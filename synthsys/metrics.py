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

"""Systematicity metrics"""

import pandas as pd


def compute_majority_vote(
  df: pd.DataFrame,
  group_cols: list[str] | None = None,
) -> pd.DataFrame:
  """Calculates majority vote across attempts for each task variation."""
  if group_cols is None:
    group_cols = ['set_id', 'variation_type', 'variation_idx']

  # Extract a single boolean outcome per attempt
  df_attempt = (
    df.groupby([*group_cols, 'attempt_idx'], dropna=False)
    .agg(correct_all=('correct_all', 'first'))
    .reset_index()
  )

  df_majority = (
    df_attempt.groupby(group_cols, dropna=False)['correct_all']
    .mean()
    .gt(0.5)
    .reset_index(name='pass_majority')
  )
  return df_majority


def compute_systematicity_metrics(df: pd.DataFrame) -> pd.DataFrame:
  """
  Calculates systematicity metrics:
  - pass_any: proportion of sets where model solves at least 1 variation
  - pass_frac: average fraction of variations solved per set
  - pass_all: proportion of sets where model solves all variations

  Returns a summary DataFrame containing per-variation metrics and an overall average.
  """
  group_cols = ['set_id', 'variation_type', 'variation_idx']
  df_majority = compute_majority_vote(df, group_cols)

  variation_types = [
    v for v in df_majority['variation_type'].unique() if v != 'canonical'
  ]

  results = []
  for var_type in variation_types:
    df_var = df_majority[df_majority['variation_type'].isin(['canonical', var_type])]

    set_metrics = (
      df_var.groupby('set_id')['pass_majority']
      .agg(pass_all='all', pass_any='any', pass_frac='mean')
      .astype(float)
      .mean()
    )

    results.append(
      {
        'Variation': var_type,
        'pass_any': set_metrics['pass_any'],
        'pass_frac': set_metrics['pass_frac'],
        'pass_all': set_metrics['pass_all'],
      }
    )

  summary_df = pd.DataFrame(results)
  macro_row = pd.DataFrame(
    [
      {
        'Variation': 'Overall (Macro)',
        'pass_any': summary_df['pass_any'].mean(),
        'pass_frac': summary_df['pass_frac'].mean(),
        'pass_all': summary_df['pass_all'].mean(),
      }
    ]
  )
  summary_df = pd.concat([summary_df, macro_row], ignore_index=True)

  return summary_df
