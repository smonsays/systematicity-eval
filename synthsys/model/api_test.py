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

import asyncio
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
from pydantic import BaseModel

from synthsys.model.api import AsyncAPIModel


class Dog(BaseModel):
  name: str
  age: int


class ModelAPITestCase(parameterized.TestCase):
  # @absltest.skip('Skipping long-running sample response test')
  @parameterized.parameters(
    dict(
      model_name='gemini-3.1-flash-lite-preview',
      provider='gemini',
      max_concurrent=3,
      reasoning_effort='low',
    )
  )
  def test_multiturn_conversation_with_structured_output(
    self,
    model_name: str,
    provider: str,
    max_concurrent: int,
    reasoning_effort: Literal['low'] | Literal['high'],
  ) -> None:
    """
    Test that async multitrun conversations work as intended by asking about multiple
    dogs like this:

    prompt("Describe a nice dog", schema=Dog)
    prompt("Describe another nice dog older than the previous", schema=Dog)
    prompt("Describe another nice dog older than the previous", schema=Dog)
    """
    model = AsyncAPIModel(
      model_name,
      provider=provider,
      max_concurrent=max_concurrent,
      reasoning_effort=reasoning_effort,
      temperature=None,
    )
    system_prompt = 'You are a dog connaisseur.'
    prompts = [
      'Describe a nice dog',
      'Describe another nice dog older than the previous',
      'Describe another nice dog older than the previous',
      'Describe another nice dog older than the previous',
      'Describe another nice dog older than the previous',
    ]
    responses = asyncio.run(model.chat([prompts], [5 * [Dog]], system_prompt))[0]

    self.assertLen(responses, 5)
    for dog in responses:
      self.assertIsInstance(dog, Dog)

    for i in range(1, len(responses)):
      self.assertGreater(
        responses[i].age,
        responses[i - 1].age,
        msg=(
          f'Dog at index {i} (age {responses[i].age}) is not older than'
          f' dog at index {i - 1} (age {responses[i - 1].age})'
        ),
      )


if __name__ == '__main__':
  absltest.main()
