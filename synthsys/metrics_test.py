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

import pandas as pd
from absl.testing import absltest

from synthsys.metrics import compute_majority_vote
from synthsys.metrics import compute_systematicity_metrics


class MetricsTestCase(absltest.TestCase):
  def test_systematicity_metrics(self) -> None:
    # Construct synthetic evaluation dataframe:
    # 2 sets (set_0, set_1)
    # 2 variations: translation (idx 1, 2) and shuffle (idx 1, 2) + canonical (idx 0)
    # 3 attempts per task
    rows = []
    # set_0: passes canonical and translation, fails shuffle
    # set_1: passes canonical, fails translation and shuffle
    for set_id, outcomes in [
      (
        'set_0',
        {
          ('canonical', 0): True,
          ('translation', 1): True,
          ('translation', 2): True,
          ('shuffle', 1): True,
          ('shuffle', 2): False,
        },
      ),
      (
        'set_1',
        {
          ('canonical', 0): True,
          ('translation', 1): False,
          ('translation', 2): False,
          ('shuffle', 1): False,
          ('shuffle', 2): False,
        },
      ),
    ]:
      for (var_type, var_idx), passed in outcomes.items():
        for attempt_idx in range(3):
          # Majority vote will match `passed`
          correct = passed if attempt_idx < 2 else not passed
          rows.append(
            {
              'set_id': set_id,
              'variation_type': var_type,
              'variation_idx': var_idx,
              'attempt_idx': attempt_idx,
              'correct_all': correct,
            }
          )

    df = pd.DataFrame(rows)

    df_majority = compute_majority_vote(df)
    self.assertEqual(len(df_majority), 10)  # 5 variations * 2 sets

    metrics = compute_systematicity_metrics(df)
    self.assertEqual(len(metrics), 3)  # translation, shuffle, overall

    trans_row = metrics[metrics['Variation'] == 'translation'].iloc[0]
    # set_0 translation: canonical(T), trans1(T), trans2(T) -> 3/3 = 1.0 frac, pass_all=1.0, pass_any=1.0
    # set_1 translation: canonical(T), trans1(F), trans2(F) -> 1/3 = 0.333 frac, pass_all=0.0, pass_any=1.0
    # mean pass_any = 1.0, mean pass_all = 0.5, mean pass_frac = (1.0 + 1/3) / 2 = 2/3
    self.assertAlmostEqual(trans_row['pass_any'], 1.0)
    self.assertAlmostEqual(trans_row['pass_all'], 0.5)
    self.assertAlmostEqual(trans_row['pass_frac'], 2 / 3)

    shuffle_row = metrics[metrics['Variation'] == 'shuffle'].iloc[0]
    # set_0 shuffle: canonical(T), shuf1(T), shuf2(F) -> 2/3 frac, pass_all=0.0, pass_any=1.0
    # set_1 shuffle: canonical(T), shuf1(F), shuf2(F) -> 1/3 frac, pass_all=0.0, pass_any=1.0
    # mean pass_any = 1.0, mean pass_all = 0.0, mean pass_frac = 0.5
    self.assertAlmostEqual(shuffle_row['pass_any'], 1.0)
    self.assertAlmostEqual(shuffle_row['pass_all'], 0.0)
    self.assertAlmostEqual(shuffle_row['pass_frac'], 0.5)

    overall_row = metrics[metrics['Variation'] == 'Overall (Macro)'].iloc[0]
    self.assertAlmostEqual(overall_row['pass_any'], 1.0)
    self.assertAlmostEqual(overall_row['pass_all'], 0.25)
    self.assertAlmostEqual(overall_row['pass_frac'], (2 / 3 + 0.5) / 2)


if __name__ == '__main__':
  absltest.main()
