# COASTLINE

This package contains ado custom experiments that expose the **COASTLINE**
recommender: *a better AutoConf that uses AutoConf*.

Where the [`autoconf`](../autoconf) plugin answers a feasibility question — "what
is the minimum number of GPUs that will run this tuning job without an OOM?" —
COASTLINE **reuses that same AutoConf check** and goes one step further: it sweeps
candidate GPU layouts, keeps the feasible ones (per AutoConf), simulates each with
the Kavier analytical throughput + power model, and returns the *best* layout under
a chosen performance/energy policy.

## Experiments

This package registers two custom experiments under the `custom_experiments`
actuator:

### `coastline_recommender` (multi-objective — headline)

Recommends the best GPU layout for a tuning job by ranking AutoConf-feasible
candidates under a preset that weights the runtime-vs-power trade-off:

- `balanced` (default), `energy`, `performance`
- `balanced-frontier`, `energy-frontier`, `performance-frontier` (rank only over
  the non-dominated Pareto frontier)

### `coastline_min_gpu_recommender` (minimum-GPU — AutoConf analogue)

Returns the smallest feasible GPU layout over the same AutoConf-gated feasible set,
reported with Kavier predicted throughput and power. This is the direct
COASTLINE-driven analogue of autoconf's `min_gpu_recommender`.

## Inputs

| Property            | Required | Default      | Notes                                                            |
| ------------------- | -------- | ------------ | ---------------------------------------------------------------- |
| `model_name`        | yes      | —            | Performance (Kavier) model                                       |
| `method`            | yes      | —            | `full` or `lora`                                                 |
| `gpu_model`         | yes      | —            | e.g. `NVIDIA-A100-80GB-PCIe`                                     |
| `tokens_per_sample` | yes      | —            | Max sequence length                                             |
| `batch_size`        | yes      | —            | Held fixed; only the GPU layout is optimised                    |
| `max_gpus`          | no       | `8`          | Sweep GPU counts (powers of two) up to this                    |
| `gpus_per_node`     | no       | `8`          | Node packing limit                                              |
| `preset`            | no       | `balanced`   | (`coastline_recommender` only)                                  |
| `feasibility`       | no       | `autoconf`   | `autoconf`, `rules`, or `none`                                  |
| `feasibility_model` | no       | `""`         | AutoConf model name override; empty → use `model_name`          |

## Outputs

`can_recommend`, `gpus` (per worker/node), `workers` (nodes), `total_gpus`,
`recommended_batch_size`, `predicted_throughput`, `predicted_power_watts`,
`predicted_runtime_seconds`, `tokens_per_watt`, `strategy`, and
`feasibility_backend` (the backend actually used — `autoconf` or, if AutoConf was
unavailable, `rules`).

## How it uses AutoConf

`feasibility="autoconf"` (the default) routes each candidate layout through the
COASTLINE feasibility checker, which loads the AutoConf rule-based + AutoGluon OOM
classifier from the `ado-autoconf` plugin. Only feasible candidates proceed to
performance/energy scoring. If `ado-autoconf` (and its AutoGluon/torch stack) is
not installed, the plugin logs a warning and degrades to a divisibility-rule check
(`feasibility_backend` reports `rules`).

## Installation

From the root of the ado repository:

```bash
# Base install (analytical recommender + divisibility-rule feasibility)
uv pip install -e plugins/custom_experiments/coastline

# Add the AutoConf feasibility backend (pulls AutoGluon/torch)
uv pip install -e "plugins/custom_experiments/coastline[autoconf]"
```

The recommendation engine itself is the `coastline-recommender` distribution, which
exposes the stable public `coastline` facade (`coastline.Coastline` /
`coastline.recommend`). This plugin talks to that facade, not to pipeline internals.

- **Published:** once `coastline-recommender` is on PyPI it is pulled in as a normal
  dependency and a plain `import coastline` is used.
- **Local dev:** install the checkout editable —
  `pip install -e <umbrella>/coastline` (and its engine `pip install -e <umbrella>/kavier`)
  — into the same venv. Note both declare `requires-python >= 3.11`.
- **Umbrella fallback:** if `coastline-recommender` is not installed, the plugin
  injects a sibling `coastline` checkout (and its `kavier` engine) onto `sys.path`
  automatically when `coastline` is a sibling of `ado`. Set `COASTLINE_ROOT`
  (and `KAVIER_ROOT`) to override the locations.

## Usage

### CLI (`run_experiment`)

```bash
run_experiment plugins/custom_experiments/coastline/examples/simple.yaml
```

Example result (`granite-3.3-8b`, lora, A100-80GB-PCIe, 2048 tokens, batch 16,
balanced):

```text
can_recommend            1
gpus                     4
workers                  1
total_gpus               4
predicted_throughput     10874.26
predicted_power_watts    177.58
tokens_per_watt          61.24
strategy                 coastline_balanced
feasibility_backend      autoconf
```

COASTLINE recommends 1 worker with 4 GPUs as the balanced perf/energy pick — a
configuration AutoConf judged feasible.

### Parameter sweep over a space

```bash
ado create space -f plugins/custom_experiments/coastline/examples/sweep/space.yaml
ado create operation -f plugins/custom_experiments/coastline/examples/sweep/operation.yaml --use-latest space
ado show entities --use-latest space -o csv --output-file coastline-entities.csv
```

Inspect the `total_gpus`, `predicted_throughput`, `predicted_power_watts` and
`tokens_per_watt` columns to compare layouts across the space.

> Note: ado's `run_experiment` initialises Ray with `runtime_env={"working_dir":
> None}`. With some Ray + uv combinations the uv runtime-env hook rejects the
> `None` working dir; if you hit `path_or_uri must be a string, got NoneType`,
> run via the venv entrypoint with the hook disabled
> (`env -u RAY_RUNTIME_ENV_HOOK ./.venv/bin/run_experiment ...`). This affects the
> `autoconf` plugin identically and is unrelated to COASTLINE.
