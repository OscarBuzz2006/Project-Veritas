# Recorded live runs

Real calls to `openai/gpt-6-luna` through OpenRouter (pinned to the OpenAI provider, no fallbacks).
These are the runs behind the report: every number in it can be recomputed from these logs without
calling the model.

| Folder | What it is | Where it appears in the report | Calls | Cost |
| --- | --- | --- | --- | --- |
| `numeric-final-1114/` | **Main run** (`configs/numeric_main.yaml`, reasoning effort `low`, `--current-date 2026-11-14`, 11 V3 questions x 30 agents x 6 rounds x 3 replicates across C1/C2/C3). 17,820/17,820 rows, no misses | Main results; "Reproducing our results" | 17,820 | see `budget.json` |
| `numeric-final-luna-none/` | **Reasoning-off run** (same as above with `--reasoning-effort none`). 17,820/17,820 rows, 20 misses (0.1%), no hidden reasoning tokens. Same starting crowds as `numeric-final-1114/`, so the two pair directly | Reasoning-off results and statistics appendix | 18,119 | $0.66 |
| `selection-v3/` | Cold question selection on the 37-candidate pool (37 questions x 8 asks), date line on | Questions, selection and true values appendix | 296 | $0.0560 |
| `selection-v1/` | First cold selection run, before the date line and the "after your training data" wording | "Why the date" (44 of 296 replies said the event had not happened) | 296 | $0.0396 |
| `movability/` | Movability check on 4 calibration questions: does the model's answer follow the user's estimate or the crowd median? | Movability check appendix | 96 | $0.0213 |

## What is in each folder

- **`provider_requests.jsonl`**: every request sent and response received, one per line (the
  complete record; the other files are derived from it). Request headers and credentials are never
  written to it.
- **`budget.json`**: the request allowance ledger (`reserved_usd` is a conservative reservation,
  not the real bill; the real cost per call is in each response's `usage.cost`).
- **`numeric-final-*`**: `numeric_runs.jsonl.gz` (one row per agent-round, gzipped),
  `numeric_runs.manifest.json` (the exact run settings), `numeric_analysis.json` (contrasts, slopes,
  per-round metrics). The raw logs exceed GitHub's 100 MB limit, so the request log is gzipped; in
  `numeric-final-1114/` it is also stripped (`provider_requests.stripped.jsonl.gz`), with the
  provider's encrypted `reasoning_details[].data` blobs replaced by `[stripped]`. The unmodified
  originals are kept locally, not in git.
- **`selection-v3/`**: `selection.json` (every candidate's error, spread, replies and outcome, with
  the rule text and thresholds) and `selected_tasks.json` (the chosen questions, tagged
  `high_spread` / `low_spread`).
- **`selection-v1/`**: the request log only. The selection tool of that time threw away its
  measurements when it failed ("only 9 eligible"); later versions save `selection.json` even on
  failure.
- **`movability/`**: `movability_summary.json` (the verdict and per-question numbers) and
  `movability_replies.jsonl` (every reply with its prompts).

## Notes

- `selection-v1` and `movability` ran **before** the `--current-date` system-prompt line and prompt
  number rounding were added. Do not pool them with the final runs.
- The candidate and calibration question files are not stored here; the question text, recorded
  truth and replies for each question are inside the files above.
- Offline demo output (`make check`, `make smoke`) is not part of this record and is gitignored
  (`results/smoke*`).
