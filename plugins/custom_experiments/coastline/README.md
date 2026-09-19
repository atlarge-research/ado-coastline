# COASTLINE

This package contains ado custom experiments that expose the **COASTLINE**
recommender: *a better AutoConf that uses AutoConf*.

Where the [`autoconf`](../autoconf) plugin answers a feasibility question —
"what is the minimum number of GPUs that will run this tuning job without an
OOM?" — COASTLINE **reuses that same AutoConf check** and goes one step
further: it sweeps candidate GPU layouts, keeps the feasible ones (per
AutoConf), simulates each with the Kavier analytical throughput and power
model, and returns the *best* layout under a chosen performance/energy policy.

The plugin is a thin adapter. It declares the experiments, validates their
inputs against ado property domains, and calls the public `coastline.Coastline`
facade of the `coastline-recommender` distribution; it never reaches into
recommender internals.

## Experiments

Both experiments are registered under the `custom_experiments` actuator.

### `coastline_recommender` (multi-objective — headline)

Recommends the best GPU layout for a tuning job by ranking AutoConf-feasible
candidates under a preset that weights the runtime-versus-power trade-off:

- `balanced` (default), `energy`, `performance`
- `balanced-frontier`, `energy-frontier`, `performance-frontier` (rank only
  over the non-dominated Pareto frontier)

### `coastline_min_gpu_recommender` (minimum-GPU — AutoConf analogue)

Returns the smallest feasible GPU layout over the same AutoConf-gated feasible
set, reported with Kavier predicted throughput and power. This is the direct
COASTLINE-driven analogue of autoconf's `min_gpu_recommender`.

## Inputs

Both experiments take the same properties, except `preset`, which only
`coastline_recommender` accepts.

| Property | Required | Default | Notes |
| --- | --- | --- | --- |
| `model_name` | yes | — | Kavier performance model: `granite-3-8b`, `granite-3.1-8b-instruct`, `granite-3.3-8b`, `llama3.2-3b`, `mistral-7b-v0.1` |
| `method` | yes | — | `full` or `lora` |
| `gpu_model` | yes | — | `L40S`, `NVIDIA-A100-80GB-PCIe`, `NVIDIA-A100-SXM4-80GB`, `NVIDIA-H100-PCIe` |
| `tokens_per_sample` | yes | — | Maximum sequence length |
| `batch_size` | yes | — | Per-device batch; held fixed, only the GPU layout is optimised |
| `max_gpus` | no | `8` | Sweep total GPU counts (powers of two) up to this |
| `gpus_per_node` | no | `8` | Node packing limit |
| `preset` | no | `balanced` | `coastline_recommender` only |
| `feasibility` | no | `autoconf` | Which mechanism gates a candidate: `autoconf`, `rules` or `none`. They are different checks, not three strengths of one — see [Feasibility](#feasibility) |
| `feasibility_model` | no | `""` | AutoConf model-name override for the OOM check; empty means use `model_name` |

Two details are easy to get wrong:

- `batch_size` is **per device**, the COASTLINE/Kavier convention. The
  `autoconf` plugin's `batch_size` is the *total* effective batch. COASTLINE
  converts at the AutoConf boundary (`effective = per_device × total_gpus`),
  so the same number describes different jobs in the two plugins.
- `feasibility_model` exists because the performance model can be a proxy for
  the model you actually tune. Set it to the AutoConf-known name when the two
  differ; leave it empty to check `model_name` itself.

## Outputs

Every prediction below describes the **whole job**, never one GPU.

| Output | Meaning |
| --- | --- |
| `can_recommend` | Whether a feasible layout was found (`1` or `0`) |
| `gpus` | GPUs per worker |
| `workers` | Workers (nodes) |
| `total_gpus` | `gpus × workers` |
| `recommended_batch_size` | Per-device batch of the winning candidate; equals the requested `batch_size`, which is held fixed |
| `predicted_throughput` | Kavier predicted throughput of the job as a whole, across all `total_gpus` GPUs [tokens/s] |
| `predicted_power_watts` | Kavier predicted power of the job as a whole: the instantaneous draw of all `total_gpus` GPUs together, **not** the per-GPU figure [W] |
| `tokens_per_watt` | `predicted_throughput / predicted_power_watts`, both job-level, so a job-level efficiency [tokens/J] |
| `strategy` | `coastline_<preset>`, or `coastline_min_gpu` |
| `feasibility_backend` | The feasibility backend that actually ran: `autoconf`, `rules` or `none` |

The COASTLINE engine's own power model reports watts *per GPU*; these
experiments multiply by `total_gpus` before publishing. A power figure here is
therefore larger than the engine's raw one for the same layout by exactly that
factor — the engine's ranking already accounts for it.

When no candidate is feasible, or an input fails validation, the experiment
reports `can_recommend` 0 and no layout fields — the same convention
autoconf's `min_gpu_recommender` follows. See
[Reading a `can_recommend` 0 row](#reading-a-can_recommend-0-row) for what can
put a row there.

There is deliberately no predicted runtime: runtime is dataset-size dependent
and these experiments take no dataset-size input (no samples, epochs, steps or
total tokens), so there is nothing from which to compute one.

## Feasibility

`feasibility` selects *which mechanism* decides whether a candidate layout is
allowed. The three values are three different mechanisms, not three strengths
of one, so rows produced under different backends have to be read separately.

| `feasibility` | What gates a candidate |
| --- | --- |
| `autoconf` (default) | The engine's structural guards (`total_gpus` ≥ 1, per-device `batch_size` ≥ 1), then AutoConf's OOM classifier — no empirical budget on top |
| `rules` | Those same structural guards **plus** an empirical per-device token budget |
| `none` | Nothing |

### `autoconf`

Routes every candidate layout through COASTLINE's feasibility checker, which
applies the engine's structural guards — a positive GPU count, a per-device
batch of at least 1 — and then the rule-based plus AutoGluon OOM classifier
loaded from `ado-autoconf`. Only survivors are scored for performance and
energy, so a recommendation is always a layout AutoConf judged feasible.

What is *not* layered on top is the empirical token budget below, deliberately:
the classifier behind those guards is the gate COASTLINE's own experiments were
run with, and leaving it alone is what keeps these rows comparable with them.

`ado-autoconf` is a dependency of `coastline-recommender`, so the `coastline`
extra below pulls it in. There is no separate AutoConf extra to install.

### `rules`

The structural guards plus COASTLINE's empirical per-device token budget: a
candidate is refused when its `batch_size × tokens_per_sample` exceeds 60,224
tokens per device, a threshold fitted to the OOMs observed in COASTLINE's
calibration campaigns. The guard can only turn feasible into infeasible.

Both factors of that product are fixed across the sweep — `batch_size` is held
fixed by design and `tokens_per_sample` is an input — so every candidate in the
grid carries the same per-device load: the budget vetoes the whole job or none
of it. It is a job-level veto, never a way of preferring one layout to another.

This is **stricter than the engine's own `feasibility="rules"`**, which is the
structural guards alone and carries no memory model of any kind. The plugin
switches the empirical budget on for this backend so that a run without
AutoConf still refuses the per-device loads that were observed to run out of
memory.

It is a much coarser check than AutoConf's. It sees the per-device token load
and nothing else — not the model, not the tuning method, not the GPU — so read
it as a floor under loads already known to fail, never as an OOM prediction.

### `none`

No check at all: every candidate in the grid is scored and ranked. The
empirical budget is not applied here either.

### When AutoConf is unavailable

If the AutoConf backend cannot be loaded at measurement time, the plugin logs a
warning and degrades to `rules`, empirical budget included, rather than
failing. The degradation is **reported, not silent**: every affected result
carries `feasibility_backend: rules`, including rows that produced no
recommendation at all.

A degraded run is therefore still memory-checked, just by the coarser
mechanism. Filter on `feasibility_backend` before comparing rows or drawing
conclusions from a sweep: a candidate refused under `autoconf` was judged by
the classifier, one refused under `rules` by the token budget, and a `rules`
layout that was accepted was never put to the classifier at all. The throughput
and power figures beside it are ordinary Kavier predictions either way.

Ask for the coarser check deliberately with `feasibility: rules`, or drop
feasibility filtering entirely with `feasibility: none`.

### Reading a `can_recommend` 0 row

`can_recommend` 0 says only that the experiment published no layout. Four
different things produce it, and the row alone does not say which:

- AutoConf's OOM classifier refused every candidate (`autoconf`).
- The empirical per-device token budget refused every candidate (`rules`) —
  with the batch size held fixed, that veto is all-or-nothing for the job.
- Nothing was scorable: an unknown or uncalibrated `model_name` makes the
  Kavier performance model return an error-carrying prediction for every
  candidate, so none survives the sweep. Nothing was judged infeasible here;
  there was simply nothing left to rank.
- The inputs never reached a verdict: a value the engine rejects makes the
  recommendation fail validation, and the plugin reports the same
  `can_recommend` 0.

`feasibility_backend` narrows this to the mechanism that was in play — it is
reported on empty rows too — but it does not separate a veto from a missing
prediction or from invalid input. Both failure branches log a warning naming
the offending inputs and the reason: the first three cases as "produced no
recommendation", the last as "invalid input". Read the log when the
distinction matters.

The degradation warning is the one gap in that log. It is emitted once per
degraded measurement, except when the measurement fails *before* the backend
is resolved (an input that cannot be coerced to an integer is rejected
first), in which case it is not emitted at all even though the row still
carries `feasibility_backend: rules`.

## Installation

From the root of the ado repository:

```bash
# The plugin: experiment definitions, input validation, ado registration
uv pip install -e plugins/custom_experiments/coastline

# The plugin and the recommendation engine
uv pip install -e "plugins/custom_experiments/coastline[coastline]"
```

The `coastline` extra installs the engine, **`coastline-recommender` 0.2.3 or
newer**. That floor is load-bearing: 0.2.3 is the release that exposes
`Coastline(empirical_oom_guard=True)` on the public facade, which is how this
plugin puts the empirical per-device token budget under the `rules` backend.

Confirm the experiments registered:

```bash
uv run ado get experiments
```

Both are listed under the `custom_experiments` actuator, at version `1.0.0`
(one row each, other columns and rows elided):

```text
coastline_min_gpu_recommender   1.0.0
coastline_recommender           1.0.0
```

### Python version

The plugin supports Python 3.11 to 3.13, as does the recommendation engine
(`coastline-recommender`, published on PyPI). Both therefore run in ado's
default 3.11 environment, and the `coastline` extra needs no environment
marker. The `<3.14` cap is inherited: the engine depends on `ado-autoconf`,
hence on `autogluon` 1.5, which caps `pyarrow<21`.

The engine is nevertheless an *extra* rather than a hard dependency, because it
pulls the autogluon/torch stack — no ado user should have to install that
merely to see these experiments in the catalog. Without the extra the plugin
still installs, registers both experiments and validates their inputs; only an
actual measurement fails, with a typed `CoastlineUnavailableError` naming the
install command.

## Usage

### 1. CLI (`run_experiment`)

Here is [examples/simple.yaml](examples/simple.yaml):

```yaml
entity:
  model_name: granite-3.3-8b
  method: lora
  gpu_model: NVIDIA-A100-80GB-PCIe
  tokens_per_sample: 2048
  batch_size: 16
  preset: balanced
  max_gpus: 8
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: coastline_recommender
    experimentVersion: 1.0.0
```

From the root directory of the ado repository, run:

```bash
uv run run_experiment plugins/custom_experiments/coastline/examples/simple.yaml
```

After a few seconds the result reports this experiment's outputs:

```text
can_recommend            1
gpus                     4
workers                  1
total_gpus               4
recommended_batch_size   16
predicted_throughput     11375.07
predicted_power_watts    638.23
tokens_per_watt          17.82
strategy                 coastline_balanced
feasibility_backend      autoconf
```

COASTLINE recommends 1 worker with 4 GPUs as the balanced performance/energy
pick — a configuration AutoConf judged feasible. Both predictions describe that
whole 4-GPU job: `predicted_throughput` is its total token rate and
`predicted_power_watts` is what its four GPUs draw together, so
`tokens_per_watt` is the job's efficiency, not one GPU's.

[examples/min_gpu.yaml](examples/min_gpu.yaml) runs the minimum-GPU experiment
the same way.

### 2. Parameter sweep over a space

This obtains recommendations for every point of a configuration space, without
benchmarking any of them.

<!-- markdownlint-disable line-length -->

```bash
uv run ado create space -f plugins/custom_experiments/coastline/examples/sweep/space.yaml
uv run ado create operation -f plugins/custom_experiments/coastline/examples/sweep/operation.yaml --use-latest space
uv run ado show measurements space --use-latest -o csv --output-file coastline-sweep.csv
```

<!-- markdownlint-enable line-length -->

Compare layouts across the space with the `total_gpus`,
`predicted_throughput`, `predicted_power_watts` and `tokens_per_watt` columns,
and check `feasibility_backend` to see which rows AutoConf actually gated.

Learn more about exploring spaces in the ado documentation for taking a
[RandomWalk on a space](https://ibm.github.io/ado/latest/user-guide/examples/tutorials/random-walk/).

## Tests

From the root of the ado repository:

```bash
uv run pytest plugins/custom_experiments/coastline/tests/
```

The suite is split by what it needs to run:

- `tests/test_experiment_registration.py` never drives the engine, so it runs
  wherever ado runs. It checks the catalog entries, input domains, defaults and
  version against this README, and it calls the experiments through their
  `@custom_experiment`-**decorated** wrappers — the same entry point ado uses —
  so that ado's entity-domain validation is covered rather than bypassed.
- `tests/test_recommender_experiment.py` exercises the recommendation itself
  through each experiment's undecorated `_original_func`, which needs no Ray,
  actuator or measurement pipeline. It does need the engine: the module skips
  itself when the `coastline` extra is absent, which is the case in ado's CI,
  where the extra is deliberately not installed. Most of its cases ask for
  `feasibility: rules`, which keeps them off the classifier — no AutoConf model
  is loaded and no prediction is made — but not off the AutoGluon/torch stack:
  `ado-autoconf` is a core dependency of `coastline-recommender`, not an extra,
  so installing the engine installs it. The cases that assert the AutoConf path
  itself skip when AutoConf will not load.
