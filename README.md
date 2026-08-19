# VISTA-Skill

VISTA-Skill implements evidence-decoupled visual transition credit assignment
for reliable evolution of procedural skills under partial observability. The
implementation lives in the standalone `vista_skill/` package. It reuses
EmbodiedBench through adapters and does not modify `EmbodiedBench/`.

The P0 method follows the latest design in
`context4agent/markdown/20260806-VISTA-Skill最新方案-证据解耦视觉转移信用分配.md`:

1. Compile a skill-predicted transition from the fixed action schema and the
   active five-field skill.
2. Build an evidence-supported transition from pre/post observations and
   public environment feedback in an isolated branch.
3. Attribute predicate-level mismatches to `belief_refresh`, `skill_update`,
   or `abstain`, then locate a skill field only for skill updates.
4. Require independent recurrence, a bounded field patch, and staged paired
   validation before a persistent skill version is promoted.

## Quick validation

```bash
python -m pytest
```

The core package uses only the Python standard library. Install the optional
OpenAI-compatible model client with `pip install -e '.[models]'`. The EB-Habitat runner
must be launched in an environment where EmbodiedBench and Habitat are already
installed:

```bash
python -m vista_skill.integrations.embodiedbench.cli --help
```

The CLI separates `experiment` from frozen `evaluate`. Full mode fails closed
without an explicit method-model backend; `rule_only` is labeled as a
diagnostic ablation. The checked-in task manifest fixes all 100 coordinates and
the dataset hash. Three independent evolution runs rotate the 60/20/20 roles,
evolve after each acquisition episode, freeze separately, and run post-hoc
paired update audits on their own held-out roles. Final evaluation rejects
non-frozen, tampered, or protocol-incompatible Skill artifacts.

## Model serving (server deployment)

The executor and the teacher are the same frozen `Qwen3-VL-8B-Instruct`,
served by vLLM as an OpenAI-compatible API.
`scripts/serve_qwen3vl_vllm.sh` is the pinned serving recipe. It runs on the
GPU server, which may be a different host from the simulator/client machine
(the recommended layout for a single-GPU client).

Pinned software contract — do not deviate without recording a protocol
deviation. Two verified stacks, picked by driver version; the serving script
itself is stack-agnostic.

**Stack A — current default (driver >= 575, CUDA 13)**, as deployed in this
machine's `max_vllm` env, probe-verified 6/6 on 2026-08-19:

| component | requirement | reason |
|---|---|---|
| vLLM | 0.27.1 | CUDA-13 wheels; will NOT run on driver 550 |
| torch | 2.13.0 | verified wheel of the vLLM 0.27 stack |
| transformers | 5.15.0 | the "NOT 5.x" constraint below is vLLM-0.11-specific |
| flashinfer-python | 0.6.16.post3 | vLLM 0.27 routes top-k/top-p sampling through it; its JIT build needs the curand fix below |
| conda CUDA toolchain | 13.0 (`cuda-nvcc`, `cuda-cudart-dev`, `cuda-cccl`) + `ninja` | flashinfer JIT-compiles its sampling kernels with this nvcc on first launch |
| weights | `Qwen3-VL-8B-Instruct` in the local HF cache | the script exports `HF_HUB_OFFLINE=1` and refuses to start otherwise |

**Stack B — legacy fallback (driver 550 => CUDA 12.4 only)**, the original
verified recipe:

| component | requirement | reason |
|---|---|---|
| vLLM | 0.11.0 | minimum with the Qwen3-VL arch on CUDA 12 |
| torch | 2.8 (cu128 on driver >= 550, cu129 on >= 575) | verified wheels |
| transformers | 4.57.x (NOT 5.x) | 5.x removed `all_special_tokens_extended`, which vLLM 0.11 reads |
| flashinfer | standalone package must be UNinstalled | import-time `TypeError` against vLLM 0.11; the bundled `vllm-flashinfer` sampler is unaffected |
| ninja | installed in the serving env | the non-eager compile path shells out to it; the script puts the env `bin/` on PATH itself |

Serving is fixed to the controlled-protocol contract:
`--max-model-len 16384` (matches `configs/vista_p0.json`; do NOT shrink it —
the 10-shot executor prompt plus `max_tokens` overflows 8192 and exhausts the
planner's retries, aborting episodes), fp8 precision, client-side
temperature 0, and support for the OpenAI `seed` field (deterministic
replays; validated by the probe below).

Launch inside tmux (long-running):

```bash
bash scripts/serve_qwen3vl_vllm.sh 8000    # 8000 matches the CLI examples; default is 8001
```

The script auto-detects the GPU layout and needs no per-host editing:

- **TP**: 1 visible card -> `tp=1`; >= 2 cards -> `tp=2`. Override with
  `TP=N`; `CUDA_VISIBLE_DEVICES` restricts detection to the listed cards.
- **Shared cards** (any card already holding > `BUSY_THRESHOLD_MIB`, default
  2048 MiB, of other processes) -> conservative profile: util <= 0.6,
  `--enforce-eager`, `--max-num-batched-tokens 4096`.
- **Dedicated single card** -> util 0.87: the measured non-KV overhead of
  this model at tp=1 non-eager is ~18.6 GiB, so util 0.87 keeps the KV cache
  above the 2.25 GiB needed for one 16384-token sequence while leaving ~3 GiB
  free for a co-resident habitat-sim on the same card.
- **Dedicated multi-GPU** -> util 0.9.

Env overrides: `VLLM_PY` (serving python; default
`/root/miniconda3/envs/max_vllm/bin/python`), `TP`, `UTIL`,
`BUSY_THRESHOLD_MIB`. The server binds `0.0.0.0` (vLLM default) with no
authentication — restrict at the firewall on shared networks.

**Protocol note**: run artifacts record the executor name/type and the tensor
parallel size. Every compared method and seed must be served by the same host
with the same settings for the whole study; never mix local/remote or
fp8/bf16 arms.

### Deployment troubleshooting (Stack A)

Failures actually hit on this machine, most recent first. #2 and #3 are
handled automatically by the serving script; #1 is fixed at the conda-env
level (activate.d hook) because it must take effect before any Python import.
Use this section to diagnose the same symptoms outside the script (bare
`python`, `docker exec`, fresh hosts).

1. **Startup aborts with `ImportError:
   /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found
   (required by <env>/lib/libicui18n.so.78)`**, raised deep inside
   `import sqlite3` via vLLM's structured-output import chain. The system
   libstdc++ (Ubuntu 22.04, gcc 12) predates `CXXABI_1.3.15`, but the env's
   ICU 78 needs it; some wheel in vLLM's full import chain loads the system
   libstdc++ first, and once any `libstdc++.so.6` is resident, libicui18n
   resolves its symbols against that old copy. The failure is load-order
   sensitive, which is misleading: bare `python -c "import sqlite3"` and even
   `import vllm` pass (vLLM's `__init__` is lazy); only the full
   `vllm.entrypoints.openai.api_server` import chain trips it. Fix applied on
   this host, at env level rather than in the script because it must precede
   every import: copy conda base's newer `libstdc++.so.6.0.34` into
   `<env>/lib/libstdc++.so.6`, and inject
   `LD_PRELOAD=<env>/lib/libstdc++.so.6` via
   `<env>/etc/conda/activate.d/00-libstdcxx.sh` (with a deactivate.d that
   strips it again), so any `conda activate max_vllm` shell is covered
   regardless of how vLLM is launched. Alternatives: `conda install -n
   max_vllm -c conda-forge libstdcxx-ng>=13`, or export the same
   `LD_PRELOAD` from the serving script. Stack-agnostic: it is an OS/env
   layout issue, independent of the vLLM version.
2. **Startup aborts with `Ninja build failed ... fatal error: curand.h: No
   such file or directory`** (Stack A only). The engine loads weights and
   captures CUDA graphs fine, then EngineCore dies during warmup: vLLM 0.27's
   sampler calls flashinfer, which JIT-compiles its sampling kernels with the
   env's conda `nvcc` — but the conda CUDA 13 packages ship no `curand.h`
   (there is no `cuda-curand-dev` in the env). The header exists only inside
   the `nvidia_curand` pip wheel (pulled in by vLLM) under
   `site-packages/nvidia/cu13/{include,lib}`, which is off nvcc's default
   search path. Fix, baked into the script: export
   `CPATH=<env>/lib/python3.12/site-packages/nvidia/cu13/include` and
   `LIBRARY_PATH`/`LD_LIBRARY_PATH` pointing at the sibling `lib/`.
   Alternatives if the wheel layout ever changes:
   `conda install -n max_vllm -c conda-forge cuda-curand-dev`, or
   `VLLM_USE_FLASHINFER_SAMPLER=0` to fall back to vLLM's native (slightly
   slower) sampler. The compiled module is cached under
   `~/.cache/flashinfer/<ver>/<cc>/cached_ops/sampling/`; it only recompiles
   after the cache is cleared or flashinfer is upgraded. Note the wheel only
   ships `libcurand.so.10` (no dev symlink) and the built module has no
   runtime dependency on it — curand is header-only here — so no manual
   symlinking is needed.
3. **`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
   doesn't exist`** when triggering the same JIT outside the script. flashinfer
   resolves nvcc from `PATH`/`CUDA_HOME`; the conda env has no
   `/usr/local/cuda`. Activate the env, or export `PATH=<env>/bin:$PATH` and
   `CUDA_HOME=<env>` like the script does.

## Client setup and experiment verification

On the simulator/client host, activate the EmbodiedBench environment
(`max_embench` for `eb-hab`) and point it at the server:

```bash
export remote_url=http://<SERVER_IP>:8000/v1             # executor (RemoteModel + CLI default)
export VISTA_METHOD_BASE_URL=http://<SERVER_IP>:8000/v1  # teacher/method backend
export OPENAI_API_KEY=EMPTY
```

Verify the serving contract before any experiment run — 6/6 checks must pass
(model list, completion + usage, seed determinism, strict `json_schema`,
trajectory reflection schema, multimodal image input):

```bash
PYTHONPATH=. python scripts/probe_vllm_endpoint.py --base-url http://<SERVER_IP>:8000/v1
```

Verification ladder, cheapest first:

1. **Unit suite** (no GPU, no network): `python -m pytest` from the repo
   root — 167 tests exercising the method with deterministic fakes for the
   model and simulator ports.
2. **Simulator smoke** (client-local GPU): from `EmbodiedBench/`, run
   `python -m embodiedbench.envs.eb_habitat.EBHabEnv` (feed `-1` on stdin to
   exit after reset) — exercises EGL context creation and scene load, the
   historically fragile step on headless hosts.
3. **Real-model smoke** (recommended after every fresh server deployment):
   `PYTHONPATH=. python scripts/vista_real_model_smoke.py` and
   `PYTHONPATH=. python scripts/vista_gate_rollout_smoke.py` with
   `VISTA_METHOD_BASE_URL` set.
4. **Full experiment and frozen evaluation**:

```bash
PYTHONPATH=EmbodiedBench:. python -m vista_skill.integrations.embodiedbench.cli \
  experiment --method full \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --executor-base-url http://<SERVER_IP>:8000/v1 \
  --method-base-url http://<SERVER_IP>:8000/v1

PYTHONPATH=EmbodiedBench:. python -m vista_skill.integrations.embodiedbench.cli \
  evaluate --mode frozen_skill \
  --skill running/vista_skill/full/seed_0/frozen_skill.json --stage audit
```

Outputs land under `running/`; every invocation requires a fresh
`--output-dir`. A six-method pilot harness is available as
`PYTHONPATH=. python scripts/pilot_6methods.py` (it honours
`VISTA_METHOD_BASE_URL` and `VISTA_EMBENCH_PY`).

See `docs/implementation.md` for architecture, invariants, experiment split
requirements, and integration details. Every experiment campaign is recorded in
`docs/experiment_log.md` (settings, results, artifacts, bug ledger) — append
new entries there rather than editing old ones.
