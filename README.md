# Tadpole

*An LLM agent built from scratch, and grown one stage at a time.*

A tool-using agent in about 3,500 lines of plain Python that you can run around the clock from
Telegram, and a stage-by-stage record of how it was built, measured and post-trained, from the
first agent loop to SFT, expert iteration and a GRPO pilot on a small (4B) model.

It is two things:

- **An agent you can use.** It reads, writes and edits files in a granted workspace, runs shell
  commands, searches and fetches the web, keeps sessions and compacts long conversations. It runs on a **local
  small model** or on **any OpenAI-compatible model** (OpenAI, DeepSeek, OpenRouter, Ollama, LM
  Studio, vLLM), in a terminal or as a Telegram bot.
- **A from-scratch tutorial.** Built without LangChain, LlamaIndex or any other agent framework:
  the loop, tools, context management and training code are plain Python you can read end to
  end. Each stage adds one mechanism with tests and measured evidence: the loop, sessions,
  tools, context management, a benchmark with an independent verifier, then post-training the
  model behind it.

| Post-training Qwen3-4B-Instruct-2507, pass@1 on task types never trained on | Test (140) | Hard test (80) | BFCL multi-turn |
|---|---:|---:|---:|
| Base | 0.419 | 0.275 | *0.360* |
| **SFT-1**: SFT on verified teacher trajectories and public tool-use data | **0.774** | 0.429 | 0.285 |
| **SFT-2**: SFT-1's data plus multi-turn demonstrations | 0.738 | **0.492** | 0.325 |
| **EI-3**: SFT-2 plus 3 rounds of expert iteration | *0.745* | *0.471* | **0.395** |

Details, the GRPO pilot and the limitations: [Post-training results](#post-training-results).

## Use the agent

### Install

Python 3.11 or later.

```sh
git clone https://github.com/shakewingo/Tadpole.git
cd Tadpole
python3 -m venv .venv && source .venv/bin/activate
pip install -r agent_from_scratch/requirements.txt
```

Run every command below from the repository root with the venv active.

### Choose a model

**Any OpenAI-compatible model.** Set three variables:

```sh
export AGENT_BASE_URL=https://api.deepseek.com   # the API root; often ends in /v1
export AGENT_MODEL=deepseek-flash                # the provider's model ID
export AGENT_API_KEY=sk-...                      # not needed for a local server
export AGENT_N_CTX=32768                         # optional: the model's context window
```

**Qwen3-4B on your own machine.** Either serve it with Ollama (any OS) and point the variables
above at it:

```sh
ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
OLLAMA_CONTEXT_LENGTH=32768 ollama serve   # Ollama's default window is too small for the agent
export AGENT_BASE_URL=http://localhost:11434/v1
export AGENT_MODEL=hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
export AGENT_N_CTX=32768
```

or, on a Mac, load the GGUF in-process with llama.cpp (about 2.5 GB, Metal-accelerated; 16 GB of
memory is plenty) and leave `AGENT_BASE_URL` unset:

```sh
hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf --local-dir models
```

That is the default location; set `AGENT_MODEL_PATH` to use another GGUF file.

A 4B model is a capable tool user on short tasks and a weak planner on long ones. For daily use,
a larger API model gives better answers; the 4B is what this project trains.

### Chat in the terminal

```sh
python -m agent_from_scratch.agent
```

### Chat from Telegram

1. In Telegram, message [@BotFather](https://t.me/BotFather), send `/newbot`, and copy the
   **bot token**. Then message [@userinfobot](https://t.me/userinfobot) and copy your numeric
   **user ID**.
2. Put both in the bot's env file, with your model variables if you use an API:

   ```sh
   cp agent_from_scratch/deploy/telegram-bot.env.example agent_from_scratch/deploy/telegram-bot.env
   # edit it: TELEGRAM_BOT_TOKEN=...  TELEGRAM_ALLOWED_USER_ID=...  (and AGENT_* if used)
   ```

3. Start it, and message your bot:

   ```sh
   sh agent_from_scratch/scripts/run_telegram_bot.sh
   ```

> [!WARNING]
> The agent runs shell commands and edits files with your user's permissions. The bot answers
> only `TELEGRAM_ALLOWED_USER_ID` and ignores everyone else. Keep the env file private: anyone
> with the token controls the bot. Run it under an account or on a machine you are prepared to
> let it change.

Commands, in Telegram and in the terminal:

| Command | What it does |
|---|---|
| `/new` | start a fresh session |
| `/reset` | clear the current session's history |
| `/session <id>` | switch to a named session |
| `/compact` | summarize the history before your next message |

### Keep it running 24/7

The bot is one long-running process, so it needs an always-on machine: a Mac mini or desktop
that does not sleep, a home server, or a small cloud VM with an API model (a 1 GB VM is enough).

**Linux (systemd).** Save as `~/.config/systemd/user/tadpole-bot.service`, with your path:

```ini
[Unit]
Description=Tadpole Telegram bot
After=network-online.target

[Service]
ExecStart=/bin/sh /path/to/Tadpole/agent_from_scratch/scripts/run_telegram_bot.sh
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

```sh
systemctl --user enable --now tadpole-bot
sudo loginctl enable-linger "$USER"     # keep it running after you log out
journalctl --user -u tadpole-bot -f   # logs
```

**macOS (launchd).**

```sh
PLIST=~/Library/LaunchAgents/com.tadpole.telegram-bot.plist
cp agent_from_scratch/deploy/com.tadpole.telegram-bot.plist "$PLIST"
sed -i '' "s|/REPLACE/WITH/REPO/PATH|$PWD|g" "$PLIST"
mkdir -p outputs && launchctl load "$PLIST"
tail -f outputs/telegram_bot.log
```

`launchctl unload "$PLIST"` stops it; unload and load again after changing code. In System
Settings, turn on "Prevent automatic sleeping when the display is off".

**Anywhere else**, `tmux` or `nohup` keeps it running until the machine restarts.

## Learn how it was built

Every mechanism was added one stage at a time, with a deliberate failure, focused tests (596 of
them, none needing a model) and evidence from real model runs.

| Stage | What you build | Code | Read and run |
|---|---|---|---|
| 0.5 | One request: model → tool call → result → answer, with validated calls and errors fed back | `llm.py`, `agent.py`, `tools/base.py` | [notebook §1](agent_from_scratch/examples/walkthrough.ipynb) |
| 1 | Multi-turn sessions, structured traces, explicit stop reasons | `session.py`, `trace.py` | notebook §2 |
| 2 | Files, shell, web search and fetch tools behind one registry | `tools/` | notebook §2 |
| 3 | Context budgeting, compaction, a summary checkpoint that survives restart | `context.py`, `compact.py` | notebook §3 |
| 8 | A generated benchmark, an independent verifier, a frozen protocol | `evals/bench/` | notebook §4, [datasets](agent_from_scratch/evals/README.md#datasets), [add a task type](agent_from_scratch/evals/README.md#adding-a-task-type) |
| 9 | SFT: verified trajectories → examples → LoRA adapter | `evals/sft/` | notebook §5–6, [SFT data](agent_from_scratch/evals/README.md#building-sft-data), [training](agent_from_scratch/evals/README.md#training-an-adapter) |
| 10 | Expert iteration and a GRPO pilot | `evals/sft/grpo.py` | notebook §7, [EI and GRPO](agent_from_scratch/evals/README.md#expert-iteration-and-grpo) |
| 11 | One final comparison on held-out tests | `evals/report.py` | [results](agent_from_scratch/docs/results.md) |

Where to start:

- **[The walkthrough notebook](agent_from_scratch/examples/walkthrough.ipynb)**: one section per
  stage, with outputs saved so it reads on GitHub. Sections 1–3 and 6 run live in minutes with
  an API model; 4, 5 and 7 need no model. Section 6 shows a teacher's run becoming SFT data, and
  section 7 expert iteration and GRPO on the 4B student's own recorded samples.
- **[The build log](agent_from_scratch/docs/STAGE.md)**: each stage's goal, what was built,
  what the evidence showed, and where the build departed from the plan.
- **[Evaluating and post-training](agent_from_scratch/evals/README.md)**: the datasets, how to
  add a task type, export SFT data, train an adapter, and run expert iteration and GRPO.

Training needs a CUDA GPU (24 GB or more). We rented one per session on RunPod; the whole
post-training line cost about \$140.

## Post-training results

The agent's code, prompts and tools were frozen; only the model's weights changed. Every choice
was made on val, and each final model was run once on two held-out sets: a test set of 140
single-turn tasks and a hard test of 80 leveled tasks, whose types no training set contains,
three samples per task.

| | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| Test pass@1 | 0.419 | **0.774** | 0.738 | *0.745* |
| Test pass^3 (all three samples pass) | 0.364 | **0.707** | 0.607 | *0.657* |
| Hard test pass@1 | 0.275 | 0.429 | **0.492** | *0.471* |
| Hard test pass^3 | 0.125 | 0.250 | *0.325* | **0.338** |
| BFCL multi-turn† | *0.360* | 0.285 | 0.325 | **0.395** |

**Bold** is the best model on a row, *italic* the second. SFT-1 is SFT on verified teacher
trajectories for 320 single-turn tasks plus When2Call and Toucan rows; SFT-2 adds teacher
trajectories for 100 more single-turn and multi-turn tasks; EI-3 is SFT-2 after three rounds of
expert iteration. † A local adaptation of BFCL's multi-turn base category, not a leaderboard
score.

**What we found**

- **SFT on verified demonstrations gives the large gain**: from 0.419 to 0.774 on test, with false
  "done" claims falling from 31 to 1 in 420 runs.
- **Three post-training routes reach the same level on held-out tasks.** Over all 220 tasks,
  SFT-1, SFT-2 and EI-3 score 0.648, 0.648 and 0.645; no difference between them is
  distinguishable from zero.
- **EI-3 is the most balanced model**: second on both held-out sets, the most reliable on the hard
  test, and first on BFCL, where it beats its starting point SFT-2 (26 entries won, 12 lost,
  p = 0.034). Public When2Call data had taught SFT-1 to ask for information it already had in
  multi-turn conversations; multi-turn demonstrations and expert iteration recover that.
- **Val overstated expert iteration.** It raised val from 0.752 to 0.852, but on held-out tasks
  EI-3 is level with SFT-2: the best of several noisy val scores is biased upwards.
- **GRPO lost to expert iteration** at every equal number of rollouts on val: few tasks per step
  had mixed outcomes, and nothing anchored the model to its other skills.
- **Neither method learned a behaviour the model never produced.** Outcome learning selects
  among what the model already samples.

**Limitations.** One training run per model. A small local benchmark (140 + 80 tasks), not a
claim about general agent ability. Base and SFT-1 ran on test in an earlier serving session than
SFT-2 and EI-3. When2Call scores are in-distribution for every model trained on public data.
Full tables, paired tests and intervals: [docs/results.md](agent_from_scratch/docs/results.md).

## Repository layout

```
agent_from_scratch/
  agent.py  llm.py  context.py  compact.py  session.py  trace.py   the agent
  tools/        files, shell, web, search, calculator, plan
  bots/         the Telegram bot
  prompts/      system prompt, compaction prompt, chat template
  evals/        benchmark, verifier, SFT, expert iteration, GRPO, external tests
  examples/     the walkthrough notebook
  docs/         the build log and the results
  tests/        unit tests (no model needed)
```

## Roadmap

Tadpole keeps growing, one measured stage at a time:

- **Memory**: durable memory across sessions, with a bounded index and on-demand detail,
  explicit remember, search and forget tools, and corrections that win over old facts;
  measured as memory off against memory on.
- **Skills and MCP**: a skill catalogue whose instructions load only when needed, and tools from
  MCP servers behind the same registry.
- **Background jobs and a sub-agent**: scheduled tasks for the always-on bot, and one read-only
  child agent.
- **A self-improving harness**: with the model fixed, let execution feedback improve the harness
  itself, starting with reflective prompt search and moving to keep-or-revert search over
  prompts and tools, scored on task types the search never saw.

Details: [the build log](agent_from_scratch/docs/STAGE.md#what-comes-next).

## Acknowledgements

The tool implementations adapt parts of [nanobot](https://github.com/HKUDS/nanobot)
(see [the attribution and license](agent_from_scratch/tools/THIRD_PARTY.md)). Thanks to the authors of
[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[When2Call](https://huggingface.co/datasets/nvidia/When2Call),
[Toucan](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) and
[BFCL](https://github.com/ShishirPatil/gorilla).

## License

Tadpole's original code and documentation are licensed under the [MIT License](LICENSE).
Copyright (c) 2026 shakewingo.

Third-party material retains its original license: the adapted nanobot tools carry their
MIT notice, and the Qwen-derived chat template is covered by Apache-2.0. See
[Third-party notices](THIRD_PARTY_NOTICES.md) for attribution, license texts and the separate
terms for model weights and external datasets.
