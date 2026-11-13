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
| `live-smoke.yaml` | The suite config used for the two smoke runs | | |

## What is in each folder

- **`provider_requests.jsonl`**: every request sent and response received, one per line (the
  complete record; the other files are derived from it). About 79% of its size is the provider's
  encrypted reasoning blobs (`reasoning_details[].data`), which are random-looking text and not
  readable. Request headers and credentials are never written to it.
- **`budget.json`**: the request allowance ledger (`reserved_usd` is a conservative reservation,
  not the real bill; the real cost per call is in each response's `usage.cost`).
- **`live-smoke-*`**: `numeric_runs.jsonl` (one row per agent-round), `numeric_runs.manifest.json`
  (the exact run settings), `numeric_analysis.json` (contrasts, slopes, per-round metrics).
- **`selection-v3/`**: `selection.json` (every candidate's error, spread, replies and outcome, with
  the rule text and thresholds) and `selected_tasks.json` (the 10 chosen questions, tagged
  `high_spread` / `low_spread`).
- **`movability/`**: `movability_summary.json` (the verdict and per-question numbers) and
  `movability_replies.jsonl` (every reply with its prompts).

## Read this before comparing runs

- `live-smoke-1`, `live-smoke-2` and `selection-v1` were run **before** the current-date line was
  added to the shared system prompt, so their prompts differ from the later runs.
- `selection-v1` kept only the request log: the selection tool of that time threw away its
  measurements when it failed ("only 9 eligible"). Later versions save `selection.json` even on
  failure.
- The candidate and calibration question files are not stored here; the question text, recorded
  truth and replies for each question are inside the files above.
- Offline demo output (`make check`, `make smoke`) is not part of this record and is gitignored
  (`results/smoke*`).
