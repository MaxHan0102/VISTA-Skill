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
deviation:

| component | requirement | reason |
|---|---|---|
| vLLM | 0.11.0 | minimum with the Qwen3-VL arch; 0.27.x requires CUDA 13 |
| torch | 2.8 (cu128 on driver >= 550, cu129 on >= 575) | verified wheels |
| transformers | 4.57.x (NOT 5.x) | 5.x removed `all_special_tokens_extended`, which vLLM 0.11 reads |
| flashinfer | standalone package must be UNinstalled | import-time `TypeError`; the bundled `vllm-flashinfer` sampler is unaffected |
| ninja | installed in the serving env | the non-eager compile path shells out to it; the script puts the env `bin/` on PATH itself |
| weights | `Qwen3-VL-8B-Instruct` in the local HF cache | the script exports `HF_HUB_OFFLINE=1` and refuses to start otherwise |

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
requirements, and integration details.
