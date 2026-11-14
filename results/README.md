# Recorded live runs

Real calls to `openai/gpt-6-luna` through OpenRouter (pinned to the OpenAI provider, low reasoning
effort), made on 13 Nov 2026. Total cost of everything here: about **$0.16**. These are the
evidence behind the question-bank decisions in issue #16; nothing here is a final experiment result.

| Folder | What it is | Calls | Cost |
| --- | --- | --- | --- |
| `live-smoke-1/` | Numeric core (C1/C2/C3), 2 questions (Timbuktu census, Yarlung Tsangpo length), 8 agents x 2 rounds x 2 replicates | 192 | $0.0203 |
| `live-smoke-2/` | The same setup again, as a repeat | 192 | $0.0212 |
| `selection-v1/` | Cold question selection on the first candidate pool (37 questions x 8 asks) | 296 | $0.0396 |
| `selection-v3/` | Cold question selection on the v3 pool (37 questions x 8 asks), stratified rule, date line on | 296 | $0.0560 |
| `movability/` | Movability check on 4 calibration questions: does the model's answer follow the user's estimate or the crowd median? | 96 | $0.0213 |
| `numeric-main/` | **Partial** unrounded main run (`configs/numeric_main.yaml`, 11 V3 questions x 30 agents x 6 rounds x 3 replicates on `openai/gpt-6-luna`). Stopped by hand during C1: 4,320 of 5,940 C1 agent-rounds done (1 miss), C2 and C3 not started | 4,409 | about $1.14 |
| `numeric-main-luna-rounded-c1/` | **Paired rounded C1 run** (`configs/numeric_main.yaml` with `format_prompt_number`, `MAX_LOG10_JUMP=2.0`, `openai/gpt-6-luna` pinned to `OpenAI`). Preserved 1,670 C1 agent-rounds across `ag_ind_gold` (540), `ag_ind_total` (540), `cwg_aus_gold` (540), and `cwg_ind_total` (50) before OpenRouter ran out of credits (`0/1,670` misses). Pairs 1-to-1 with the first 1,670 rows of `numeric-main/` to measure the prompt-formatting effect directly | 1,672 | $0.3295 |
| `live-smoke-3/` | **Rounded V3 smoke run** (`results/live-smoke-3.yaml`, `openai/gpt-6-luna` pinned to `OpenAI`, `ag_ind_total` + `box_odyssey_cume`, 8 agents x 2 rounds x 1 replicate across C1/C2/C3). Verified `smoke-report`: `0/96` misses, `96/96` reasoning tokens recorded, `0` truncated, `max_identity_residual = 1.28e-16` | 96 | $0.0206 |
| `live-smoke-adaption/` | **Rounded V3 smoke run on Adaption** (`results/live-smoke-3.yaml` on `deepseek-v4-flash` via `--backend adaption`, 8 agents x 2 rounds x 1 replicate across C1/C2/C3). Verified `smoke-report`: `0/96` misses, `96/96` reasoning tokens recorded, `0` truncated, `max_identity_residual = 1.16e-16` | 100 | $0.0518 |
| `numeric-main-rounded/` | **Complete 11-question V3 rounded run on Adaption** (`configs/numeric_v3_adaption.yaml`, all 11 V3 questions across `high_spread`, `low_spread`, and `control` strata, 6 agents x 2 rounds x 1 replicate across C1/C2/C3 on `gemma-4-26b-a4b-it`). `396/396` rows (`0/396` misses), `max_identity_residual = 5.83e-16` | 396 | $0.0550 |
| `numeric-v3-qwen397b/` | **Complete 11-question V3 rounded run on Adaption (`qwen3.5-397b`)** (`configs/numeric_v3_adaption.yaml`, all 11 V3 questions, 6 agents x 2 rounds x 1 replicate across C1/C2/C3 with `reasoning={"effort": "none"}`). `396/396` rows (`0/396` misses, `0` truncated), `max_identity_residual = 4.86e-16` | 510 | $2.5007 |
| `live-suite-adaption/` | **Live multi-model Track 4 + V3 numeric suite on Adaption** (`configs/adaption_live_suite.yaml`: 5 Track 4 conditions including 4-model `cross_national_k4` mixing `deepseek-v4-flash`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `qwen3.8-27b`, 80 Track 4 rows + 24 V3 numeric rows on `gemma-4-31b-it`, `0/24` numeric misses, `max_identity_residual = 1.67e-16`) | 136 | $0.0182 |
| `live-smoke.yaml` | The suite config used for the two earlier unrounded smoke runs (`live-smoke-1`, `live-smoke-2`) | | |
| `live-smoke-3.yaml` | The V3 rounded smoke suite config + `live-smoke-3-tasks.json` | | |

## What is in each folder

- **`provider_requests.jsonl`**: every request sent and response received, one per line (the
  complete record; the other files are derived from it). About 79% of its size is the provider's
  encrypted reasoning blobs (`reasoning_details[].data`), which are random-looking text and not
  readable. Request headers and credentials are never written to it.
- **`budget.json`**: the request allowance ledger (`reserved_usd` is a conservative reservation,
  not the real bill; the real cost per call is in each response's `usage.cost`).
- **`numeric-main/`**: same file layout as `live-smoke-*` (no `numeric_analysis.json`: the run did not finish).
  The run was stopped early, so it cannot be resumed into a complete result without care: resume with the
  same command and settings and the saved rows are reused.
- **`live-smoke-*`**: `numeric_runs.jsonl` (one row per agent-round), `numeric_runs.manifest.json`
  (the exact run settings), `numeric_analysis.json` (contrasts, slopes, per-round metrics).
- **`selection-v3/`**: `selection.json` (every candidate's error, spread, replies and outcome, with
  the rule text and thresholds) and `selected_tasks.json` (the 10 chosen questions, tagged
  `high_spread` / `low_spread`).
- **`movability/`**: `movability_summary.json` (the verdict and per-question numbers) and
  `movability_replies.jsonl` (every reply with its prompts).

## Read this before comparing runs

- **Pilot vs V3 separation:** `live-smoke-1`, `live-smoke-2`, `selection-v1`, and `movability` belong to the earlier question-bank selection/pilot phase and ran **before** the `--current-date` system-prompt line and prompt number rounding were added. Do not pool them with V3 rounded runs (`live-smoke-3`, `numeric-main-luna-rounded-c1`, `live-smoke-adaption`, `numeric-main-rounded`, `numeric-v3-qwen397b`, `live-suite-adaption`).
- **Paired formatting-effect comparison (`openai/gpt-6-luna`, `N = 1,670` matched C1 rows):** `numeric-main/` (unrounded decimals/scientific notation) vs `numeric-main-luna-rounded-c1/` (3 sig figs rounded to whole numbers with commas) both had `0/1,670` misses on the matched cells; H4 `slope_user` measured `0.1006` `[0.0482, 0.1530]` unrounded vs `0.1104` `[0.0502, 0.1706]` rounded.
- **Mechanical convergence baseline ($\alpha = 0.3$):** Under `agent_after = (1 - alpha) * agent_before + alpha * assistant`, an uncoupled constant assistant contracts within-crowd variance by $(1 - \alpha)^2 = 0.49$ per round regardless of social influence. In `numeric-main-rounded` (`gemma-4-26b-a4b-it`, 11 V3 questions), round-1 to round-2 diversity ratios were `0.589`/`0.684` (`C1`), `0.555`/`0.590` (`C2`), and `0.595`/`0.683` (`C3`), while `C2`/`C3` showed higher final crowd error (`0.0511` / `0.0525` vs `0.0252` in `C1`) and higher `slope_user` (`0.1695` / `0.2611` vs `0.0920` in `C1`). In `numeric-v3-qwen397b` (`qwen3.5-397b`, 11 V3 questions), round-2/round-1 diversity ratios were `0.7169` (`C1`), `0.6661` (`C2`), and `0.6918` (`C3`); final crowd error rose from `0.0255` (`C1`) to `0.0302` (`C2`) and `0.0313` (`C3`), and `qwen3.5-397b` exhibited stronger baseline user-echoing (`slope_user = 0.3069` `C1`, `0.2414` `C2`, `0.2723` `C3`, `h4_assistant_agent_corr = 0.9985`).
- `selection-v1` kept only the request log: the selection tool of that time threw away its
  measurements when it failed ("only 9 eligible"). Later versions save `selection.json` even on
  failure.
- The candidate and calibration question files are not stored here; the question text, recorded
  truth and replies for each question are inside the files above.
- Offline demo output (`make check`, `make smoke`) is not part of this record and is gitignored
  (`results/smoke*`).
