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

"""Asynchronous LLM API client wrapper with structured output support."""

import asyncio
import json
import logging
from typing import Any
from typing import Generic
from typing import Iterator
from typing import Literal
from typing import TypeVar

import backoff
import google.genai.errors
import httpx
import pydantic
from any_llm import acompletion
from any_llm import completion
from any_llm.exceptions import ContentFilterError
from any_llm.exceptions import ProviderError
from any_llm.exceptions import RateLimitError
from any_llm.logging import setup_logger

setup_logger(level=logging.DEBUG)

T = TypeVar('T', bound=pydantic.BaseModel)

CONTENT_ERRORS = (
  ContentFilterError,
  AttributeError,
  IndexError,
  TypeError,
  json.JSONDecodeError,
  pydantic.ValidationError,
)

API_ERRORS = (
  RateLimitError,
  ProviderError,
  httpx.ReadError,
  httpx.RemoteProtocolError,
  google.genai.errors.APIError,  # Slips through any-llm so we catch it manually
)

BACKOFF_ERRORS = (*CONTENT_ERRORS, *API_ERRORS)

ReasoningEffort = Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto']


class ResponseWithExplanation(pydantic.BaseModel, Generic[T]):
  response: T
  explanation: str


class AsyncAPIModel:
  """API models using the any-llm-sdk library."""

  def __init__(
    self,
    model_name: str,
    max_concurrent: int,
    reasoning_effort: ReasoningEffort,
    temperature: float | None,
    provider: str | None = None,
  ) -> None:
    self.model_name = model_name
    self.provider = provider
    self.semaphore = asyncio.Semaphore(max_concurrent)
    self.reasoning_effort = reasoning_effort
    self.temperature = temperature

  async def generate(
    self, prompt: str, response_schema: type[T], system_prompt: str
  ) -> T:
    async with self.semaphore:
      return await self._chat_completion_request(prompt, response_schema, system_prompt)

  @backoff.on_exception(
    backoff.expo,
    BACKOFF_ERRORS,
    max_tries=10,
    max_time=600,
    base=2,
    factor=12,
    jitter=backoff.random_jitter,
  )
  async def _chat_completion_request(
    self,
    prompt: str,
    response_schema: type[T],
    system_prompt: str,
  ) -> T:
    messages = [
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': prompt},
    ]
    response = await acompletion(
      model=self.model_name,
      provider=self.provider,
      messages=messages,
      response_format=response_schema,
      reasoning_effort=self.reasoning_effort,
      temperature=self.temperature,
    )
    return response.choices[0].message.parsed

  async def respond_maybe_with_explanation(
    self, prompt: str, response_schema: type[T], system_prompt: str, explain: bool
  ) -> ResponseWithExplanation:
    if explain:
      system_prompt += (
        'In addition to the requested response, please provide a short sentence '
        'describing the rule you identified.'
      )
      wrapper_schema = pydantic.create_model(
        'ResponseWithExplanation', explanation=(str, ...), __base__=response_schema
      )
    else:
      wrapper_schema = response_schema

    try:
      async with self.semaphore:
        result = await self._chat_completion_request(
          prompt, wrapper_schema, system_prompt
        )

      response = response_schema(**result.model_dump(exclude={'explanation'}))
      explanation = getattr(result, 'explanation', '')

    except CONTENT_ERRORS as e:
      logging.error(f'Generation failed due to: {type(e).__name__} - {e}')
      dummy_data = {name: None for name in response_schema.model_fields.keys()}
      response = response_schema.model_construct(**dummy_data)
      explanation = 'FAILED_TO_GENERATE' if explain else ''

    return ResponseWithExplanation(response=response, explanation=explanation)

  async def chat(
    self,
    prompts: list[list[str]],
    response_schemas: list[list[type[T]]],
    system_prompt: str,
  ) -> list[list[T]]:
    tasks = [
      self._chat_single_conversation(p, s, system_prompt)
      for p, s in zip(prompts, response_schemas, strict=True)
    ]
    return await asyncio.gather(*tasks)

  async def _chat_single_conversation(
    self, prompts: list[str], response_schemas: list[type[T]], system_prompt: str
  ) -> list[T]:
    async with self.semaphore:
      messages = [{'role': 'system', 'content': system_prompt}]

      # Process each turn in the conversation sequentially
      results = []
      for prompt, schema in zip(prompts, response_schemas, strict=True):
        result = await self._chat_turn_request(messages, prompt, schema)
        results.append(result)

      return results

  @backoff.on_exception(
    backoff.expo,
    BACKOFF_ERRORS,
    max_tries=10,
    max_time=600,
    base=2,
    factor=12,
    jitter=backoff.random_jitter,
  )
  async def _chat_turn_request(
    self,
    messages: list[dict[str, Any]],
    prompt: str,
    response_schema: type[T],
  ) -> T:
    # We append to a new list so that if backoff retries, it doesn't repeatedly
    # add the same prompt to the global message history list.
    current_messages = [*messages, {'role': 'user', 'content': prompt}]

    response = await acompletion(
      model=self.model_name,
      provider=self.provider,
      messages=current_messages,
      response_format=response_schema,
      reasoning_effort=self.reasoning_effort,
      temperature=self.temperature,
    )

    parsed_result = response.choices[0].message.parsed
    content = response.choices[0].message.content

    # If content is empty (common with some structured outputs), use the dumped JSON.
    if not content and parsed_result:
      content = parsed_result.model_dump_json()

    # Apply successful turn to the conversation history
    messages.append({'role': 'user', 'content': prompt})
    messages.append({'role': 'assistant', 'content': content or ''})

    return parsed_result


class APIModel:
  def __init__(
    self,
    model_name: str,
    reasoning_effort: ReasoningEffort,
    temperature: float | None,
    provider: str | None = None,
  ) -> None:
    self.model_name = model_name
    self.provider = provider
    self.reasoning_effort = reasoning_effort
    self.temperature = temperature

  def __call__(self, prompt: str, system_prompt: str) -> str:
    messages = [
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': prompt},
    ]
    response = completion(
      model=self.model_name,
      provider=self.provider,
      messages=messages,
      reasoning_effort=self.reasoning_effort,
      temperature=self.temperature,
    )
    return response.choices[0].message.content or ''

  def chat(self, prompts: list[str], system_prompt: str) -> Iterator[str]:
    messages = [{'role': 'system', 'content': system_prompt}]

    for prompt in prompts:
      messages.append({'role': 'user', 'content': prompt})
      response = completion(
        model=self.model_name,
        provider=self.provider,
        messages=messages,
        reasoning_effort=self.reasoning_effort,
        temperature=self.temperature,
      )
      content = response.choices[0].message.content or ''
      messages.append({'role': 'assistant', 'content': content})
      yield content


if __name__ == '__main__':
  model = AsyncAPIModel(
    'gemini-3.1-flash-lite-preview',
    provider='gemini',
    max_concurrent=2,
    reasoning_effort='low',
    temperature=None,
  )
  prompts = [
    'What is 3 + 6 mod 8?',
    'What is 1 + 6 mod 8?',
    'What is 3 + 6 3 mod 8?',
  ]
  system_prompt = 'You are so good at modular arithmetic, wow! Output answer and carry.'
  schema = pydantic.create_model('M', **{'answer': int, 'carry': int})

  async def eval_prompts(prompts: list[str]) -> list[dict]:
    tasks = []
    for prompt in prompts:
      task = model.generate(prompt, schema, system_prompt)
      tasks.append(task)

    return await asyncio.gather(*tasks)

  print(asyncio.run(eval_prompts(prompts)))
