# Results: post-training a 4B tool-using agent

Can a small open model become a better agent through post-training alone, with the agent's code,
prompts and tools held fixed? We trained LoRA adapters for
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) with supervised
fine-tuning (SFT), expert iteration (EI) and a GRPO pilot, and ran each final model once on task
types that no training set contains. The datasets are described in
[evals/README.md, "Datasets"](../evals/README.md#datasets).

## Summary

- **SFT on verified demonstrations gives the large gain.** On 140 single-turn test tasks of 14
  types that training never saw, pass@1 rises from 0.419 to 0.774, and false "done" claims fall
  from 31 to 1 in 420 runs.
- **Three post-training routes reach the same level on held-out tasks.** Over all 220 held-out
  tasks (test and hard test), SFT-1, SFT-2 and EI-3 score 0.648, 0.648 and 0.645; no difference
  between them is distinguishable from zero.
- **EI-3 is the most balanced model.** It is second on test (0.745) and on the hard test (0.471),
  the most reliable on the hard test (pass^3 0.338), and first on BFCL multi-turn (0.395); no
  other model is in the top two on all four. Its BFCL gain over its starting point, SFT-2, is
  significant (26 entries won, 12 lost, p = 0.034); no in-harness difference between two trained
  models is.
- **The validation set overstated expert iteration.** EI-3 raised val from 0.752 to 0.852, but on
  held-out tasks it is level with its starting point.
- **The GRPO pilot lost to expert iteration** at every equal number of rollouts on val, and was not
  run on held-out tasks.

This is a small local benchmark. None of it is a claim about general agent ability.

## The models

| Model | Trained on | Rows |
|---|---|---:|
| **Base** | Qwen3-4B-Instruct-2507, no adapter | — |
| **SFT-1** | DeepSeek's verified trajectories on the 320 single-turn train tasks (×3), plus 2,000 When2Call and 3,002 Toucan rows | 7,870 |
| **SFT-2** | SFT-1's data, with the teacher trajectories extended to 419 of the 420 single-turn and multi-turn train tasks | 9,007 |
| **EI-3** | Three rounds of expert iteration from SFT-2. Each round samples the 80 leveled train tasks 8 times with the latest model, keeps the verified rollouts, and retrains on SFT-2's data plus those rollouts | about 9,300 per round |
| **GRPO** (val only) | SFT-2's adapter, updated online: 8 samples per task, a strict pass/fail reward, group-normalised advantages, 30 steps over three passes | — |

All adapters use one recipe: LoRA rank 16 on every attention and MLP projection of the bf16
base, loss on the assistant's tokens only, AdamW at $10^{-4}$ with warmup and cosine decay, up to
three epochs. Every choice (epoch, data mix, which round to keep) was made on val. The test, the
hard test and the external test were each run once per model, after the choices were made. All
training data and every evaluation used the agent with planning off (no `update_plan` tool), as
the benchmark manifest freezes it.

## Held-out results

Three samples per task. **Bold** is the best model on a row, *italic* the second.

| | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| test pass@1 (140 single-turn tasks) | 0.419 | **0.774** | 0.738 | *0.745* |
| test pass^3 (all three samples pass) | 0.364 | **0.707** | 0.607 | *0.657* |
| hard test pass@1 (80 leveled tasks) | 0.275 | 0.429 | **0.492** | *0.471* |
| hard test pass^3 | 0.125 | 0.250 | *0.325* | **0.338** |
| all 220 held-out tasks, pass@1 | 0.367 | **0.648** | **0.648** | 0.645 |
| all 220 held-out tasks, pass^3 | 0.277 | **0.541** | 0.505 | **0.541** |
| BFCL multi-turn, local adaptation | *0.360* | 0.285 | 0.325 | **0.395** |
| When2Call, LLM judge (mean of three kinds) | 0.533 | 0.697 | *0.713* | **0.723** |
| false completions (of 660 runs) | 132 | 107 | **82** | *94* |

The "all 220 tasks" rows pool the test and the hard test; they summarize the two sets and are
not a separate measurement.

Paired comparisons, by task and seed:

| B against A | Set | B wins | B loses | McNemar p | B − A, 95% interval |
|---|---|---:|---:|---:|---|
| SFT-1 vs Base | test | 182 | 33 | <0.001 | +0.355 [+0.257, +0.450] |
| SFT-1 vs Base | hard test | 82 | 45 | 0.001 | +0.154 [+0.021, +0.287] |
| EI-3 vs SFT-2 | test | 39 | 36 | 0.82 | +0.007 [−0.043, +0.055] |
| EI-3 vs SFT-2 | hard test | 36 | 41 | 0.65 | −0.021 [−0.113, +0.071] |
| EI-3 vs SFT-1 | all 220 | 56 | 58 | 0.93 | −0.003 [−0.041, +0.035] |
| SFT-2 vs SFT-1 | all 220 | 71 | 71 | 1.00 | −0.000 [−0.044, +0.044] |

The interval resamples tasks, keeping the three samples of a task together (10,000 resamples).

## Test (140 single-turn tasks)

By task family, pass@1:

| | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| inspection (38) | 0.439 | 0.833 | 0.719 | **0.895** |
| recovery (32) | 0.688 | **1.000** | **1.000** | **1.000** |
| stopping (32) | 0.094 | **0.438** | 0.396 | 0.354 |
| updates (38) | 0.447 | 0.807 | **0.825** | 0.711 |
| passed per seed (of 140) | 57, 58, 61 | 105, 111, 109 | 109, 99, 102 | 106, 101, 106 |

<details>
<summary>Test pass@1 by task type</summary>

| Type | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| `ambiguous_choice_stop` | 0.00 | 1.00 | 1.00 | 0.72 |
| `append_list_item` | 1.00 | 0.33 | 0.94 | 0.50 |
| `backup_record_lookup` | 0.50 | 1.00 | 1.00 | 1.00 |
| `deep_chain_lookup` | 1.00 | 1.00 | 1.00 | 1.00 |
| `flaky_write_retry` | 1.00 | 1.00 | 1.00 | 1.00 |
| `grep_locate` | 0.56 | 0.72 | 0.83 | 1.00 |
| `merge_record_requests` | 0.00 | 1.00 | 0.97 | 0.90 |
| `missing_file_report` | 0.00 | 0.56 | 0.61 | 0.67 |
| `pointer_nested_edit` | 0.94 | 0.44 | 0.56 | 0.28 |
| `rename_key_all_files` | 0.89 | 1.00 | 0.50 | 0.72 |
| `sum_across_files` | 0.44 | 0.22 | 0.33 | 0.33 |
| `three_source_join` | 0.23 | 1.00 | 0.72 | 1.00 |
| `transient_read_failure` | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_complete_total` | 0.15 | 0.23 | 0.15 | 0.15 |

</details>

## Hard test (80 leveled tasks)

Four types in formats no train or val type uses, at the difficulty level calibrated on SFT-2.
pass@1:

| Type | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| `config_patch_plan` (an ordered INI patch) | 0.100 | 0.517 | **0.667** | 0.450 |
| `shipping_fee_table` (a first-match fee table) | **0.433** | 0.167 | 0.367 | 0.250 |
| `ledger_audit_fault` (an audit with a failed read) | 0.000 | 0.600 | 0.433 | **0.683** |
| `support_dialogue` (a multi-turn ticket) | **0.567** | 0.433 | 0.500 | 0.500 |
| false completions (of 240) | 101 | 106 | **82** | 91 |

Every model claims a finished task it did not finish in a third or more of its runs here, and
each type ranks the models differently: these tasks still mostly measure where all of them fail.

## External test

| | Base | SFT-1 | SFT-2 | EI-3 |
|---|---:|---:|---:|---:|
| BFCL multi-turn base (local adaptation) | 0.360 | 0.285 | 0.325 | **0.395** |
| When2Call MCQ accuracy (length-normalised) | 0.456 (0.500) | **0.694** (0.675)* | 0.691 (0.671)* | 0.675 (0.658)* |
| When2Call hallucination rate (lower is better) | 0.225 | **0.027*** | 0.039* | 0.035* |
| When2Call LLM judge, mean of three kinds | 0.533 | 0.697* | 0.713* | **0.723*** |
| judge: tool call / ask for info / cannot answer | 0.93 / 0.21 / 0.46 | 0.56 / 0.86 / 0.67 | 0.61 / 0.88 / 0.65 | 0.59 / 0.86 / 0.72 |

\* In-distribution: these models trained on rows of When2Call's train set, from the same
generator as its test set. BFCL, paired by entry: EI-3 beats SFT-2 on 26 entries and loses 12
(p = 0.034), beats SFT-1 on 34 and loses 12 (p = 0.002), and beats Base on 27 and loses 20
(p = 0.38). BFCL here is our adaptation (our chat template and decoding, one category) and is not
comparable with the public leaderboard; the When2Call judge is DeepSeek (`deepseek-flash`).

## Expert iteration against GRPO (val)

Both start from SFT-2 and sample the same 80 leveled train tasks, 8 times each, so one round of
expert iteration and one pass of GRPO see the same 640 rollouts. Val has 144 tasks: 64
single-turn, 40 multi-turn and 40 leveled.

| Model | Rollouts seen | Val (144) | Single- and multi-turn (104) | Leveled (40) | False completions |
|---|---:|---:|---:|---:|---:|
| SFT-2 (start) | 0 | 0.752 | 0.955 | 0.225 | 33 |
| EI-1 | 640 | 0.785 | 0.907 | 0.467 | 48 |
| EI-2 | 1,280 | 0.729 | 0.856 | 0.400 | 38 |
| **EI-3** | 1,920 | **0.852** | 0.942 | **0.617** | **3** |
| GRPO-1 | 640 | 0.718 | 0.891 | 0.267 | 53 |
| GRPO-2 | 1,280 | 0.722 | 0.885 | 0.300 | 51 |
| GRPO-3 | 1,920 | 0.731 | 0.875 | 0.358 | 52 |

As a data generator, each round's model solves more of the train pool (attempt success 0.58,
0.74, 0.75, 0.77), with most of the gain in round 1. GRPO raised its train reward (0.58 → 0.72)
while val fell.

## What the evidence shows

- **SFT on verified demonstrations is the gain.** Every trained model beats Base by about 0.28 over
  all held-out tasks. Most of the gain is in finishing correctly and saying so honestly: false
  completions on test fall from 31 to 0–3.
- **Public tool-use data trades one skill for another.** When2Call's "ask for the missing
  parameter" rows taught SFT-1 to ask for information it already had in multi-turn conversations:
  on BFCL it makes no tool call in 147 of 734 user turns, against Base's 26, and scores below Base.
  The multi-turn demonstrations of SFT-2, then expert iteration, recover it: of the three, EI-3
  alone scores above Base on BFCL (27 entries won, 20 lost; not significant).
- **Expert iteration adds balance, not level.** On held-out tasks EI-3 matches its starting point
  overall (75 tasks won, 77 lost) while improving outside the harness (BFCL) and on the hard
  test's reliability. Its val gain was overstated: it was chosen among seven candidates after
  rounds that were not monotone (0.785, 0.729, 0.852), and the best of several noisy val scores is
  biased upwards. Each round's self-generated data was about 3% of its training rows.
- **Each change moves skills around.** SFT-2 recovers the JSON-list edit that SFT-1 lost
  (`append_list_item` 0.33 → 0.94); EI-3 is the best on inspection (0.895) and on the audit with a
  failed read (0.683), and the weakest trained model on updates (0.711). With 6 tasks per test
  type, single-type differences are noisy.
- **GRPO at this scale did not work.** Few tasks per step had mixed outcomes (0 to 5 of 8), so each
  update rested on a handful of groups, and nothing anchored the model to its other skills: the
  single- and multi-turn val tasks fell from 0.955 to 0.875–0.891 and false completions rose.
- **Neither method learns a behaviour the model never produces.** On the leveled train type
  `long_edit_dialogue`, SFT-2 always asked and never acted correctly, so there was no success to
  select, and it stayed at 0.50 through every round and pass.

## Other models trained along the way

Two more adapters were run on the held-out sets and are not in the tables above. One was trained
on teacher trajectories for 80 single-turn tasks only, without public data (test 0.724, hard test
0.267, BFCL 0.445). The other was a variant of SFT-2 with most of When2Call's follow-up-question
rows removed (test 0.714, hard test 0.442, BFCL 0.385). Neither changes the conclusions above.

## Limitations

- One training run per model; the epochs and the round were chosen on val.
- A small local benchmark: 140 test and 80 hard-test tasks. Multi-turn behaviour is held out only
  in one hard-test type and in BFCL; there is no held-out multi-turn set at fixed difficulty.
- The hard test's levels were calibrated on SFT-2, which may favour it. One type,
  `ledger_audit_fault`, is not monotone in its level.
- Base and SFT-1 ran on test and the external test in an earlier serving session than SFT-2 and
  EI-3. Sampling on a batched server is not exactly repeatable: one adapter's greedy BFCL score
  differed by 0.025 between two sessions.
- BFCL is a local adaptation; When2Call is in-distribution for every model trained on public data.
- The agent's harness (tools, limits, prompts, planning off) is fixed. A different harness could
  rank the models differently.

## Cost

| Part | Cost |
|---|---:|
| Benchmark baseline (one A40) | \$0.35 |
| SFT: teacher data, training and the first test | about \$18 |
| Multi-turn data, expert iteration and GRPO (A100) | about \$112 |
| Final runs on test, hard test and external test | about \$10 |
| Teacher and judge API calls (DeepSeek) | a few dollars |

About \$140 in all, inside a \$200 ceiling. GPU time was rented per session on RunPod.

## Reproduce

[evals/README.md, "Reproduce the results"](../evals/README.md#reproduce-the-results) covers
setup, the frozen task set, reloading adapters, one test run and the tables.
