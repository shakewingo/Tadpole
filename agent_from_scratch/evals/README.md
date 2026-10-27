# Evaluating and post-training the agent

This folder holds everything after the agent itself: a generated benchmark with an independent
verifier, the export of verified runs as training data, a LoRA trainer, expert iteration, a
GRPO pilot, two external benchmarks and the report that builds the results tables.

- [Quick start](#quick-start)
- [Datasets](#datasets): what every set contains and which stage used it
- [Adding a task type](#adding-a-task-type)
- [Building SFT data](#building-sft-data): teacher runs, self-sampled runs, public data
- [Training an adapter](#training-an-adapter), and [on a rented GPU](#training-on-a-rented-gpu)
- [Expert iteration and GRPO](#expert-iteration-and-grpo)
- [External test](#external-test)
- [Reproduce the results](#reproduce-the-results)
- [When you change the agent](#when-you-change-the-agent)
- [Reading a run directory](#reading-a-run-directory)

## Quick start

Everything runs from the repository root through one entry point:

```sh
python -m agent_from_scratch.evals COMMAND --output outputs/NEW_NAME [options]
```

`--output` must be a new directory; evidence is never overwritten.

| I want to know… | Command | Needs | Time |
|---|---|---|---|
| Do the units still work? | `python -m unittest discover -s agent_from_scratch/tests -q` | runtime dependencies; training tests also need `requirements-train.txt`; no model | — |
| Do the task generator and verifier work? | `dataset --output DIR --check`: builds train/test tasks and runs a scripted correct solution and two cheats through the real agent loop | runtime dependencies; no model | about a minute |
| Did my change break what the agent could do? | `bench --split val` | a model | about 12 min on a 16 GB Mac with a local Qwen3-4B GGUF |
| How reliable is a model across samples? | `bench --seed N` per seed, then `aggregate DIR...`: pass@1, pass^k, pass@k, mixed-outcome tasks | a model | — |
| How do two runs differ? | `compare A B`: paired outcomes, exact McNemar p, trajectory shape | nothing | instant |
| Does context management hold when evidence outgrows the window? | `pressure`: 4 tasks over 9 files of about 15k characters | a model | 15–25 min |
| How do I turn runs into SFT data? | `export --track teacher\|rft RUN...` | nothing | seconds |
| How do I train an adapter? | `train --examples DATASET` | a CUDA GPU (24 GB+) | 17 min to 5 h |
| How do I build the results tables? | `report --arm NAME=RUN,RUN,RUN ...` | nothing | seconds |

`bench` runs one model on one split: `--backend llama_cpp` (default, a local GGUF, for pipeline
checks), `--backend vllm --base-url URL [--adapter NAME]` (the measured protocol), or
`--backend teacher` (an API model, train split only). `--tasks a,b` runs a subset.

## Datasets

### How a task is made

A **task** is a small disposable workspace (a few text or JSON files), a prompt, the tools the
agent may use, and a private contract that the agent never sees: the expected final answer, the
expected file contents, the facts the agent must have read, and what must stay unchanged. The
verifier checks the contract after the run, from the files and the trace, not from what the
agent claims.

A **task type** (called a *skeleton* or *shape* in the code) is a generator: given a seed, it
writes a new workspace with different values, file names, distractors and fault positions, and
the matching contract. Every type also has a scripted correct solution and two cheats (answer
without reading, and read then answer wrongly). A type is accepted only when, through the real
agent loop, the scripted solution passes and both cheats fail.

**Sets are split by type, not by task.** Two tasks of one type differ only in their values, so a
random split would put near-copies of test tasks into training. No type appears in more than one
of train, val and test, and each new type records the closest existing type and how it differs.

Every type belongs to one **family**:

| Family | What it tests | Example |
|---|---|---|
| inspection | find and combine facts across files | `three_source_join`: read `request.json` for a person, `directory.json` for their email and team, `locations.json` for the team's city; reply `email\|city` |
| updates | change exactly what was asked, nothing else | `rename_key_all_files`: rename one key in every config file under a folder |
| recovery | meet a failure and recover | `transient_read_failure`: the first read fails once; the agent must retry, not guess |
| stopping | know when not to act, or when to report | `missing_file_report`: only a backup file exists; reply `MISSING` instead of reading the backup as if it were current |

Some types come in **clean/fault pairs**: the same files, with one tool call made to fail in the
fault variant. Some come in **act/ask pairs**: the same records, with the fact needed to act
either established (in an earlier turn, a file or a tool result) or genuinely missing, where the
right reply is to ask.

### Three groups of task types

| Group | What it is | Example |
|---|---|---|
| **Single-turn** | One request, 2–6 dependent tool calls over config files and the records of a general assistant (inventories, applications, bookings, rosters, expenses) | `merge_record_requests`: merge two change requests into a contact record, or change nothing and reply `NEED_INPUT` when they conflict |
| **Multi-turn** | Follow-up user turns that refer to earlier ones ("change *that* customer's phone"), in act/ask pairs | `followup_same_record`: turn 1 names a customer, or only counts customers in a city; turn 2 changes "that customer's" phone, or replies `NEED_INPUT` |
| **Leveled** | Harder types with a difficulty level 1–5 (more records, longer rule chains, more turns); some are dialogues | `ordered_migration`: apply the N ordered steps of `plan.txt` to `data.json`, where later steps name keys that earlier steps created |

A leveled type's level is set by **calibration** on the model being trained: it is sampled 8
times on every type and level, on calibration seeds that no other set uses, and the level whose
pass rate is closest to 0.5 (inside 0.25–0.75) is kept, so that the model solves about half. A
type with no level in that band is dropped. The chosen levels are in
[`bench/levels.json`](bench/levels.json).

Most types are scored with three checks beyond the answer and the files: the facts a decision
rests on are read before the first write, a fault variant passes only if its fault was met, and
a claim failing either counts as a false completion. The 18 single-turn types of the first
benchmark (the 64 single-turn val tasks and 60 of the test tasks) keep the scoring they were
frozen with.

### The sets

| Set | Single-turn | Multi-turn | Leveled | In all |
|---|---|---|---|---|
| **train** | 380 tasks, 19 types | 40 tasks, 2 types | 80 tasks, 4 types | 500 |
| **val** (every choice is made here) | 64 tasks, 8 types | 40 tasks, 2 types | 40 tasks, 2 types | 144 |
| **test** (run once per final model) | 140 tasks, 14 types | — | — | 140 |
| **hard test** (run once per final model) | — | — | 80 tasks, 4 types | 80 |

- **Train, single-turn**: 16 record-keeping types, three to five per family, including four
  where nothing needs doing, one that reports a missing file and one that finds a file moved
  after a failed read; `lookup_then_act`, a single-turn act/ask pair; and two JSON-structure
  edits (`json_remove_array_item`, `json_nested_key_add`).
- **Train, multi-turn**: `followup_same_record`, `followup_file_from_listing`.
- **Train, leveled**: `ordered_migration` (L1), `rule_table_apply` (L1), `long_edit_dialogue`
  (L2, a dialogue), `fault_amid_reconcile` (L3).
- **Val**: `pointer_lookup`, `single_field_edit`, `check_fix_recheck`, `no_op_correct_config`,
  `max_timeout_service`, `delete_key`, `transient_list_failure`, `locked_config_stop`
  (single-turn); `followup_move_meeting`, `followup_carry_value` (multi-turn); `flag_rollout`
  (L2) and `itinerary_dialogue` (L5, a dialogue) (leveled).
- **Test**: `deep_chain_lookup`, `grep_locate`, `sum_across_files`, `pointer_nested_edit`,
  `rename_key_all_files`, `append_list_item`, `transient_read_failure`, `flaky_write_retry`,
  `ambiguous_choice_stop`, `missing_file_report` (6 tasks each), and `three_source_join`,
  `merge_record_requests`, `backup_record_lookup`, `unit_complete_total` (20 each). The test set
  was fixed before any training data existed and never changed.
- **Hard test**: `config_patch_plan` (L1, an ordered INI patch), `shipping_fee_table` (L1, a
  first-match fee table), `ledger_audit_fault` (L2, a ledger audit with a failed read) and
  `support_dialogue` (L4, a multi-turn support ticket), in formats no train or val type uses.
- **Calibration**: every leveled type at all five levels, seeds 100–105. It only chooses levels;
  it is never trained on or reported.

Generate the train and test sets, with their gates, without a model:

```sh
python -m agent_from_scratch.evals dataset --output outputs/NEW_DATA --check
```

The dataset exporter currently selects only `train` and `test` (640 tasks in total).
It does not export or gate the 144 validation or 80 hard-test tasks. Those generators are
covered by the unit suite, and `bench --split val` / `bench --split hardtest` run them with a
model. The generated manifest records all splits even though the exported workspaces do not.

Each task's `workspace/` holds only its inputs; `task.json` beside it holds the private contract.
Never send the contract or the scripted solution to a model. A scripted pass checks the
generator and verifier, not a model's ability.

### External data

| Data | Used as | Notes |
|---|---|---|
| **Teacher trajectories**: DeepSeek (`deepseek-flash`) running our agent loop on the train tasks | SFT data | only runs the verifier passed; the context budget is counted in the student's tokens, so the harness behaves as it would for the student |
| **Self-sampled trajectories**: the model being trained, 8 samples per train task | expert-iteration data, GRPO rollouts | verified runs only for expert iteration; GRPO uses every rollout of a task with mixed outcomes |
| **When2Call** `train_pref` ([nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call), CC-BY-4.0): the chosen reply of each row | SFT data | 2,000 rows: tool calls, follow-up questions and declines |
| **Toucan** SFT config (Apache-2.0) | SFT data | 3,002 examples from 966 rows, a third with tools that do not fit the request |
| **When2Call test** | external test | multiple choice (3,652 rows) and an LLM-judged free-answer set (300 rows) |
| **BFCL multi-turn base** ([gorilla](https://github.com/ShishirPatil/gorilla)) | external test | 200 entries, played in a plain chat loop on BFCL's own simulated backends and scored by its checker; a local adaptation, not a leaderboard score |

Public rows are converted into our format exactly: our system prompt, the row's tools in our
template, one example per assistant turn. A row that cannot be converted exactly is dropped and
counted, never repaired. Nothing derived from a test set is ever trained on.

### What each model was trained and chosen on

| Model | Trained on | Chosen on |
|---|---|---|
| **SFT-1** | teacher trajectories on the 320 single-turn record-keeping tasks, plus When2Call and Toucan | the 64 single-turn val tasks |
| **SFT-2** | SFT-1's data, with teacher trajectories on 419 of the 420 single-turn and multi-turn train tasks | val, single-turn and multi-turn (104 tasks) |
| **EI-3** | SFT-2's data plus its own verified rollouts on the 80 leveled train tasks, three rounds | all 144 val tasks, the leveled 40 read first |
| **GRPO** | online updates from SFT-2's adapter on the same 80 leveled tasks | all 144 val tasks; not run on held-out sets |

Every final model ran once on the test, the hard test and the external test. Val grew with the
training data: a single-turn val cannot see a model that asks instead of acting in a
conversation, and once SFT-2 solved nearly every single- and multi-turn val task, the leveled val
types gave the comparison room to move.

## Adding a task type

A type lives in [`bench/skeletons.py`](bench/skeletons.py) (single-turn and multi-turn) or
`bench/shapes_*.py` (leveled). It has three parts:

1. **A builder** `build(rng, root, ctx) -> Task` that writes the workspace under `root` from
   `rng` and returns the [`Task`](bench/spec.py): `prompt`, the allowed `tools`, and an `Expect`
   with `answer` (`"required"`: the reply must be exactly the token; `"advisory"`: the token
   must appear), `files` (expected final contents), `evidence` (strings that must appear in some
   successful tool output, i.e. what must have been read), `process` rules such as
   `no_write_attempts`, and optionally a `Fault` and `followups` (later user turns). Leveled
   builders also take `level`.
2. **A scripted `solution(task)`** that returns the model responses a correct run produces. It
   is used only by the gate tests, never as training data.
3. **Registration** with a family and a split, and the name in
   [`bench/splits.json`](bench/splits.json). A leveled type is registered only once
   `levels.json` names its level.

Then run the gates: `python -m unittest agent_from_scratch.tests.test_bench_skeletons` and
`dataset --check`. The scripted solution must pass, and answering without reading and answering
wrongly must both fail. Rules that keep the sets honest:

- Record the closest existing type in the other splits and how yours differs.
- Never add or retune a test type after looking at test results.
- Any change to a frozen type, tool, prompt, limit or the verifier needs a new freeze.

`pressure` tasks are separate: a builder in [`pressure.py`](pressure.py) fills a temporary
directory and returns `(prompt, expected_answer)`.

## Building SFT data

Two tracks collect rollouts on the train split, and `export` turns the verifier-passed ones into
`(prompt, completion)` examples, one per model request:

```sh
# Teacher: an API model inside the same loop, tools and verifier. Needs DEEPSEEK_API_KEY; the
# student GGUF only supplies the tokenizer for the context budget.
python -m agent_from_scratch.evals bench --backend teacher --split train --seed 0 \
  --student-gguf MODEL.gguf --output outputs/NEW_TEACHER_RUN
# Self-sampled: the model being trained on vLLM, one run per seed.
python -m agent_from_scratch.evals bench --backend vllm --split train --seed 3 --output outputs/NEW_RUN_S3
# One dataset per track: examples.jsonl plus metadata.json (sources and the funnel).
python -m agent_from_scratch.evals export --track teacher --output outputs/NEW_SFT outputs/NEW_TEACHER_RUN
```

The prompt is exactly what the model saw (the recorded input messages, rendered by the inference
template), and the completion is the assistant turn that followed. `export` refuses runs that
are not on the train split, scripted fixtures, and runs from another backend than the track's.
`--flawed-turns drop` keeps a verified rollout that holds the model's own failed tool call and
leaves out only that turn; the default `reject` drops the whole rollout. `--max-per-task K`
caps the trajectories kept per task.

Public data and mixing:

```sh
python -m agent_from_scratch.evals public --source when2call-pref --input when2call_train_pref.jsonl \
  --revision SHA --student-gguf MODEL.gguf --max-examples 2000 --output outputs/NEW_PUBLIC
# Our examples repeated k times, so that they come closest to the --own-share of the rows.
python -m agent_from_scratch.evals mix --own outputs/NEW_SFT --public outputs/NEW_PUBLIC ... \
  --output outputs/NEW_MIX
```

Sources: `when2call-sft`, `when2call-pref` (the chosen reply) and `toucan`. `--cap
question=200` limits one reply category.

## Training an adapter

```sh
# Needs requirements-train.txt (torch, transformers, peft), or vLLM's own stack on a GPU pod.
# One adapter per epoch, in PEFT format, plus train_log.jsonl and metadata.json.
python -m agent_from_scratch.evals train --examples outputs/NEW_SFT --output outputs/NEW_ADAPTERS
# Against a vLLM server started with --lora-modules NAME=outputs/NEW_ADAPTERS/epoch-3:
python -m agent_from_scratch.evals replay --examples outputs/NEW_SFT --adapter NAME \
  --ids outputs/NEW_ADAPTERS/metadata.json --output outputs/NEW_REPLAY.json
python -m agent_from_scratch.evals bench --backend vllm --split val --seed 0 --adapter NAME --output outputs/NEW_VAL
```

The recipe: LoRA rank 16 on every attention and MLP projection of the bf16 base; the loss falls
on the assistant's completion tokens only (the prompt, with its tool results, is context); AdamW
at $10^{-4}$, 10% warmup then cosine decay, gradient clipping at 1.0, three epochs, gradient
checkpointing on. `train` refuses a dataset exported under another chat template and a row
longer than the window (never truncated). `--limit 8 --epochs 40 --save-last` is the overfit
smoke test. `replay` checks that a served adapter reproduces its training targets greedily: a
fit check, never a capability result. Pick the epoch on val seed 0.

### Training on a rented GPU

Training and every adapter evaluation need a CUDA GPU. We rent one RunPod pod per session and
drive it from the Mac with `runpodctl`. The 4B model in bf16 needs at least 24 GB; an A40
(48 GB) serves every model and trains small datasets, and training SFT-1 or SFT-2, whose
longest rows hold 23k–30k tokens, needs an A100 80 GB.

1. **Prepare.** `runpodctl` 2.14 or later (`runpodctl doctor`), and your SSH key added with
   `runpodctl ssh add-key` before the pod is created. Commit first: the pod runs a clone of the
   last commit.
2. **Rent**, with `--min-cuda-version 13.0` (vLLM 0.30.0 installs PyTorch for CUDA 13). The pod
   cannot stop itself, so start a cost guard on your machine (`sleep 7200 && runpodctl pod stop
   POD_ID`). **Keep that machine awake and on power**: a laptop lid closed on battery freezes the
   guard and the pod bills idle.

   ```sh
   runpodctl pod create --name NAME --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
     --gpu-id "NVIDIA A40" --cloud-type SECURE --min-cuda-version 13.0 \
     --container-disk-in-gb 60 --ports "22/tcp" --wait --wait-timeout 15m
   ```

3. **Install** into a venv on the pod. Do not install `requirements-train.txt` there: vLLM pins
   its own torch and transformers, and an adapter should be trained with the stack that serves
   it. Set `OMP_NUM_THREADS=8`; a container that reports the host's 128 cores stalls the CPU
   tests otherwise.

   ```sh
   python3 -m venv /root/venv && /root/venv/bin/pip install -q uv && . /root/venv/bin/activate
   uv pip install vllm==0.30.0 peft -r agent_from_scratch/requirements.txt
   ```

4. **Train** before starting the server, which takes the GPU memory. Start long runs with
   `setsid nohup ... &`. [`scripts/train_stage9_adapters.sh`](../scripts/train_stage9_adapters.sh)
   trains the SFT adapters and lists the GPU and time each needs.
5. **Serve and evaluate** with the [server command](#the-server-command) below, then
   `bench --backend vllm --adapter NAME`.
6. **Copy back and delete.** Copy `outputs/` back and check it, then `runpodctl pod delete`, and
   confirm `runpodctl pod list --all` is empty.

### The server command

Every result was measured with this one command; use it unchanged, adding one
`--lora-modules NAME=DIR` pair per adapter:

```sh
PATH=/root/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup \
  vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --dtype bfloat16 --max-model-len 32768 --enable-lora --max-lora-rank 64 --port 8000 \
  --lora-modules SFT-1=outputs/sft-1 EI-3=outputs/ei-3 > vllm.log 2>&1 < /dev/null &
curl -s localhost:8000/v1/models   # ready when it lists the model with max_model_len 32768
```

Two settings are required. The venv's `bin/` must be on `PATH`, because FlashInfer calls `ninja`
by name. `VLLM_USE_FLASHINFER_SAMPLER=0`, because FlashInfer compiles its sampler at startup
and the image's `nvcc` (CUDA 12.8) does not match PyTorch's CUDA 13; vLLM then samples with
its PyTorch implementation, with the same decoding parameters. The served revision is not
visible through the API, so this command is what pins it.

## Expert iteration and GRPO

Both start from one SFT adapter (SFT-2 in our runs) and sample the same train pool at 8 seeds. They are compared
on val at an equal number of rollouts.

```sh
# How good a data generator is a model? `passed`, and GRPO's strict reward (passed, with the
# answer exactly as requested).
python -m agent_from_scratch.evals aggregate outputs/RUN-s0 ... outputs/RUN-s7 --strict
# One expert-iteration round: the starting model's training data plus the latest model's own
# verified rollouts, added once.
python -m agent_from_scratch.evals mix --own TEACHER --public WHEN2CALL TOUCAN --self SELF_EXPORT --output DIR
# GRPO: the task IDs of one iteration, then one optimizer step on that iteration's runs. The
# step writes the next adapter with the AdamW state beside it (optimizer.pt).
python -m agent_from_scratch.evals grpo tasks --pass 1 --iteration 0
python -m agent_from_scratch.evals grpo step --runs outputs/IT-s0 ... outputs/IT-s7 \
  --adapter-in ADAPTER_DIR --output outputs/NEXT_STEP [--lr 1e-5]
```

A GRPO row is one model request of a rollout whose task had mixed outcomes: the prompt as sent
and the text as sampled, weighted by the rollout's advantage within its group of $G = 8$:

$$A_i = \frac{r_i - \bar r}{\sigma_r}$$

The loss is the SFT loss with each row weighted by its advantage; one step per batch keeps it
on-policy, so no importance ratio or clipping is needed. With $A_i = 1$ for verified rows and the
rest dropped, it is SFT again.

On a pod, the drivers in `scripts/` run whole sessions: `stage10_sample.sh` (one model on the
train split), `stage10_ei_round.sh K` (one round) and `stage10_grpo.sh PASSES` (resumable);
`STAGE10_DRY=1` prints their commands without running them. `stage10_calibrate.sh` chooses the
leveled types' levels. [`tests/test_grpo_exploits.py`](../tests/test_grpo_exploits.py) checks
that cheap cheats earn no reward.

## External test

Run against the vLLM server like `bench` (`--adapter NAME`, or none for the base):

```sh
# When2Call: the benchmark's own multiple-choice scoring for Qwen, and its LLM-as-a-judge path
# (answers from the server, then DeepSeek classifies them; needs DEEPSEEK_API_KEY).
python -m agent_from_scratch.evals when2call mcq --data when2call_test_mcq.jsonl --output outputs/NEW_MCQ
python -m agent_from_scratch.evals when2call generate --data when2call_test_llm_judge.jsonl --output outputs/NEW_GEN
python -m agent_from_scratch.evals when2call judge --responses outputs/NEW_GEN --output outputs/NEW_JUDGE
# BFCL multi-turn base, a local adaptation, from a checkout of ShishirPatil/gorilla.
python -m agent_from_scratch.evals bfcl --bfcl-root gorilla/berkeley-function-call-leaderboard --output outputs/NEW_BFCL
```

## Reproduce the results

The numbers are in [docs/results.md](../docs/results.md). The three adapters and their SFT data
are on Hugging Face:

| Model in the results | Adapter | Directory as trained |
|---|---|---|
| SFT-1 | [Shakewin/tadpole-qwen3-4b-sft-1](https://huggingface.co/Shakewin/tadpole-qwen3-4b-sft-1) | `sft-mix-r1/epoch-2` |
| SFT-2 | [Shakewin/tadpole-qwen3-4b-sft-2](https://huggingface.co/Shakewin/tadpole-qwen3-4b-sft-2) | `sft-mix-v3-r1/epoch-2` |
| EI-3 | [Shakewin/tadpole-qwen3-4b-ei-3](https://huggingface.co/Shakewin/tadpole-qwen3-4b-ei-3) | `s10-m4` |

[Shakewin/tadpole-sft-data](https://huggingface.co/datasets/Shakewin/tadpole-sft-data) holds the
verified teacher and expert-iteration rows; its card rebuilds the converted When2Call and Toucan
rows from their sources, and each training set exactly, checked against the hashes recorded at
training time. The full measured run archives are not published, and the historical
`scripts/stage*` drivers assume local outputs from earlier sessions: they are not a fresh-clone
pipeline. The evaluation procedure is:

1. **Setup** on a GPU pod (one A40 is enough to serve): a clean clone,
   `uv pip install vllm==0.30.0 peft -r agent_from_scratch/requirements.txt`, and the adapters
   under `outputs/`, e.g. `hf download Shakewin/tadpole-qwen3-4b-ei-3 --local-dir outputs/ei-3`.
2. **Freeze the tasks and run the gates.** Every workspace is rebuilt from its seed, so there is
   no state to reset by hand:

   ```sh
   python -m agent_from_scratch.evals dataset --output outputs/NEW_DATA --check   # gates_passed: true
   ```

3. **Reload the adapters**: the [server command](#the-server-command) with one
   `--lora-modules NAME=DIR` per adapter,
   then `replay --adapter NAME` against that adapter's training examples.
4. **One test run** under the freeze (`--split hardtest` works the same way):

   ```sh
   python -m agent_from_scratch.evals bench --backend vllm --split test --final \
     --manifest outputs/NEW_DATA/manifest.json --seed 0 --adapter NAME --output outputs/NEW_TEST_S0
   ```

   `--final` refuses to run if the code, prompts, tasks or decoding differ from the freeze.
   Sampling differs between server sessions, so a rerun matches pass@1 closely, not exactly.
5. **The tables**, from one run per seed for each model:

   ```sh
   python -m agent_from_scratch.evals report --arm "A=RUN_S0,RUN_S1,RUN_S2" --arm "B=..." \
     --pair A:B [--subset NAME=TYPE,...] --output TABLE.md
   ```

   Runs are paired by the seed each run's `metadata.json` records, in any order; the report
   refuses an arm that repeats a seed, or a pair run with different seeds.

`scripts/stage11_session.sh` runs the whole final session (freeze, test, hard test, external).

## When you change the agent

1. Unit tests. They are the only check that runs without a model.
2. `bench --split val` before and after the change with the same `--seed`. Never tune against
   test: it is reserved for comparing weights.
3. `pressure` if you touched context, compaction, planning, or a tool with bulky output. `bench`
   cannot see these: its tasks stay far below the window's triggers.
4. `compare` the two runs. Read direction and trajectory, not only the p-value: with a few dozen
   tasks the exact McNemar test rarely reaches significance.

To ablate one mechanism, keep everything else fixed and pass any `AgentLimits` field as JSON;
`metadata.json` records what was on:

```sh
python -m agent_from_scratch.evals pressure --output outputs/t3 --limits '{"elide_ratio": null}'
python -m agent_from_scratch.evals pressure --output outputs/t4
python -m agent_from_scratch.evals compare outputs/t3 outputs/t4
```

## Reading a run directory

```
outputs/NAME/
  metadata.json        settings (incl. n_ctx), seed, git revision, source/task hashes, limit overrides
  results.json         one record per task: passed, checks, stop_reason, metrics()
  summary.json         pass counts by family, requests, usage
  trajectory.json      stop reasons, survival by request, action mix, mechanism use
  TASK_ID/             task.json, result.json, state/runs/*.jsonl (full trace)
```

Read first: `stop_reason` (`context_limit` is overflow; the target is zero), `false_completion`
(a reply claiming done while the checks fail), `asked_instead` (asked or declined where acting
was expected), `peak_context_ratio`, `compact_requests`, and `actions` (one label per request:
explore, modify, execute, plan, answer or error).

## What is in this folder

| Path | Role |
|---|---|
| `__main__.py` | the command dispatcher |
| `bench/` | task types (`skeletons.py`, `shapes_*.py`), splits and levels, the verifier (`verify.py`), the freeze and protocol check (`manifest.py`), `aggregate.py` |
| `sft/` | `export.py` (runs → examples), `public.py` (public rows → examples), `mix.py`, `teacher.py` (the API teacher backend), `tensorize.py` (the completion-only loss mask), `train.py` (LoRA training and the GRPO step), `replay.py`, `grpo.py` (reward, advantages, task rotation) |
| `external/` | When2Call and BFCL scorers |
| `report.py` | pass@1, paired comparisons, bootstrap intervals, the tables |
| `pressure.py` | the context-pressure suite |
| `trajectory.py` | `compare` and trajectory profiles |
| `run.py`, `verify.py` | shared run setup and post-run measurement |
