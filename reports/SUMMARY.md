# Project Veritas / Epistemic Prism (`numeric-question-bank`) Summary

- **Repository / Remote:** `https://github.com/OscarBuzz2006/Project-Veritas.git` (`project-veritas`)
- **Branch:** `numeric-question-bank` (PR #19)

## Shipped Engineering Changes
1. **Prompt Rounding & Unit-Slip Guard ([numeric_core.py](file:///Users/vikramviswanath/apart_epistemics/src/apart_epistemics/engine/numeric_core.py)):**
   - Added `format_prompt_number` (rounds to 3 significant figures, then whole integer with commas, no scientific notation) across `current_x`, `pop_stats` median/IQR (`C2`/`C3`), and `C3` history while keeping internal `agent_log10_before` / `agent_log10_after` exact.
   - Tightened `MAX_LOG10_JUMP` from `3.0` to `2.0` to reject $10^3\times$ and $10^6\times$ unit slips and trigger retry.
   - Added cross-trajectory parallelism when `workers > n_agents` while preserving sequential `workers <= n_agents` behavior and canonical record ordering.
2. **V3 Question Selection Validation ([selection.py](file:///Users/vikramviswanath/apart_epistemics/src/apart_epistemics/datasets/selection.py)):**
   - Enforced `--max-per-family` preflight capacity checks before API calls and `len(selected) < keep` checks under `rule="error"`, saving partial `selection.json` on failure.
3. **Adaption Backend & Track 4 Concurrency ([providers.py](file:///Users/vikramviswanath/apart_epistemics/src/apart_epistemics/backends/providers.py), [base.py](file:///Users/vikramviswanath/apart_epistemics/src/apart_epistemics/backends/base.py), [track4.py](file:///Users/vikramviswanath/apart_epistemics/src/apart_epistemics/engine/track4.py)):**
   - Added `AdaptionBackend` (`--backend adaption`) for `/api/v1/responses` with secret redaction, reasoning-token and cost extraction, `top_p=0.0` omission, `min_output_tokens=8192` for numeric runs, `1024` cap on short Track 4 turns, `reasoning={"effort": "none"}` on `qwen*` models (avoiding >60s AWS ALB 504 timeouts while retaining visible Fermi CoT), and jittered 60s rate-limit retry backoff (`max_retries=25`). Note `openai/gpt-6-luna` is unavailable on Adaption (`HTTP 404`).
   - Added `--workers` intra-round concurrency in Track 4 (`run_suite`) preserving deterministic `a_idx` record ordering.

## Verified Live Runs & Findings (Separated from Earlier Pilots)
- **`results/live-smoke-3/` (`openai/gpt-6-luna`, rounded V3 smoke, 96 rows):** `0/96` misses, `96/96` reasoning tokens recorded, `0` truncated, `$0.0206`, `max_identity_residual = 1.28e-16`.
- **Paired Prompt-Formatting Comparison (`openai/gpt-6-luna`, `N = 1,670` matched C1 rows across `results/numeric-main/` unrounded vs `results/numeric-main-luna-rounded-c1/` rounded):** `0/1,670` misses on matched cells; H4 `slope_user` `0.1006` `[0.0482, 0.1530]` (unrounded) vs `0.1104` `[0.0502, 0.1706]` (rounded); preserved separately before OpenRouter credits ran out (`HTTP 402`).
- **`results/live-smoke-adaption/` (`deepseek-v4-flash` on Adaption, rounded V3 smoke, 96 rows):** `0/96` misses, `96/96` reasoning tokens recorded, `0` truncated, `$0.0518`, `max_identity_residual = 1.16e-16`.
- **`results/numeric-main-rounded/` (`gemma-4-26b-a4b-it` on Adaption, all 11 V3 questions x 6 agents x 2 rounds across C1/C2/C3, 396 rows):** `0/396` misses, `$0.0550`, `max_identity_residual = 5.83e-16`. Final round metrics: `C1` crowd error `0.0252` (diversity `0.0204`), `C2` `0.0511` (`0.0166`), `C3` `0.0525` (`0.0206`); H4 `slope_user` rose from `0.0920` (`C1`) to `0.1695` (`C2`) and `0.2611` (`C3`). Mechanical contraction baseline under $\alpha = 0.3$ is $(1-\alpha)^2 = 0.49$ per round; observed round-2 ratios were `0.684` (`C1`), `0.590` (`C2`), and `0.683` (`C3`).
- **`results/numeric-v3-qwen397b/` (`qwen3.5-397b` on Adaption with `reasoning={"effort": "none"}`, all 11 V3 questions x 6 agents x 2 rounds across C1/C2/C3, 396 rows):** `0/396` misses, `0/396` truncated, `$2.5007` across `510` requests, `max_identity_residual = 4.86e-16`. Final round metrics: `C1` crowd error `0.0255` (diversity `0.0228`, round-2/round-1 diversity ratio `0.7169`), `C2` `0.0302` (`0.0202`, ratio `0.6661`), `C3` `0.0313` (`0.0207`, ratio `0.6918`). `qwen3.5-397b` showed consistently higher baseline user-following (`slope_user` `0.3069` `C1`, `0.2414` `C2`, `0.2723` `C3`; `h4_assistant_agent_corr = 0.9985`), and in `low_spread` questions social context (`C2`/`C3`) raised final crowd error from `0.0395` (`C1`) to `0.0616` (`C2`) / `0.0610` (`C3`).
- **`results/live-suite-adaption/` (`configs/adaption_live_suite.yaml`, 5 Track 4 conditions mixing `deepseek-v4-flash`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it`, `qwen3.8-27b` = 80 Track 4 rows + 24 V3 numeric rows on `gemma-4-31b-it`):** `0/24` numeric misses, `max_identity_residual = 1.67e-16`, total cost `$0.0182` across 136 requests.

## Open Risks & Next Steps
- **OpenRouter Credits (`reports/NEEDS_OWNER.md`):** OpenRouter account credits are exhausted (`HTTP 402`), halting full 17,820-row rounded sweeps on `openai/gpt-6-luna`; Adaption funded the completed 11-question V3 (`396` rows), `deepseek-v4-flash` (`96` rows), `gemma-4-31b-it` (`24` numeric rows), and 4-model Track 4 (`80` rows) experiments.
- **Full-Scale Replicates on Open Models or Topped-Up OpenRouter:** Scaling `configs/numeric_main.yaml` (30 agents x 6 rounds x 3 replicates = 17,820 rows) once OpenRouter credits or higher Adaption RPM limits are available.
