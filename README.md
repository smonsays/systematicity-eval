# Thought without systematicity? Evaluating reasoning models on rule induction tasks

Offical code for the paper [Thought without systematicity? Evaluating reasoning models on rule induction tasks]().

> A central tenet of human cognition is systematicity, the principle that understanding one concept is inherently tied to understanding close variations of that concept.
> Do reasoning models robustly exhibit such systematicity?
> If so, we would expect consistent performance on structurally equivalent variants of the same task.
> Here, we extend established rule induction tasks from cognitive science to assess the systematicity of thought in current reasoning models.
> Each task family has compositional structure that we use to create structurally equivalent task variations through task isomorphisms such as recombination and substitution.
> We find that despite being able to correctly solve a task, models often fail on structurally equivalent variants of the same task.
> These findings suggest that many model behaviors lack systematicity, rendering it difficult to robustly establish the cognitive abilities of reasoning models beyond the particular contexts they were evaluated in.

## Setup

We use [uv](https://docs.astral.sh/uv/) to handle python and dependencies, you can run `uv sync` to setup the environment.

Model provider's API credentials are assumed to be stored in a private `.env` file in the following.

## Run experiments

**Please note that experiments may incur significant API costs.**

The main experiments are defined as wandb sweeps in `sweeps/` and can be invoked as follows.

```sh
uv run wandb sweep sweeps/mlc.yaml
uv run wandb sweep sweeps/raven.yaml
uv run wandb sweep sweeps/listint.yaml
uv run wandb sweep sweeps/boolean.yaml
```

Individual tasks can be run for a single model using the configurations defined in `configs/`, for example

```sh
uv run --env-file .env run.py --config=configs/mlc.py --config.model.name 'gemini-3.1-flash-lite-preview'
uv run --env-file .env run.py --config=configs/raven.py --config.model.name 'gemini-3.1-flash-lite-preview'
uv run --env-file .env run.py --config=configs/listint.py --config.model.name 'gemini-3.1-flash-lite-preview'
uv run --env-file .env run.py --config=configs/boolean.py --config.model.name 'gemini-3.1-flash-lite-preview'
```

Each run reports systematicity metrics over configured task variations.

- `pass_any`: Solve any of $K$ variations
- `pass_frac`: Fraction of variations solved
- `pass_all`: Solve all $K$ variations

## Project structure

```text
run.py                  Evaluation entry point and result aggregation
configs/                Per-dataset experiment configurations
sweeps/                 wandb sweep definitions for main experiments
synthsys/
  config_classes.py     Typed dataset and model settings
  data/                 Task generators
  model/                Model API wrappers
  metrics.py            Systematicity metrics
```

Tests live alongside the implementation in `*_test.py` files.
