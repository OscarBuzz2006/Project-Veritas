# Sapere Aude: Does a Crowd That Asks One AI Assistant Find the Truth, or the Assistant?

**A multi-agent simulation and evaluation testbed measuring how shared LLM assistants reshape crowd diversity, calibration, and collective accuracy.**

- **Paper:** *Sapere Aude: Does a crowd that asks one AI assistant find the truth, or the assistant?* (Apart Research AI × Epistemics Sprint, November 2026)
- **Authors:** Raghav Tyagi, Kevin O'Shaughnessy, Vikram Viswanath, Oscar Burrows
---

## About

**Sapere Aude** (Latin: *"Dare to use your own wisdom"*) is an open-source research workbench for investigating how groups of AI-assisted people form beliefs, converge on estimates, and lose or preserve epistemic diversity. It pairs a reproducible Python simulation and evaluation pipeline with an interactive React interface for inspecting traces, condition comparisons, and round-by-round crowd trajectories.

The repository centers on two core components:
1. **The Numeric Crowd Testbed (`results/numeric-final-1114`, `results/numeric-final-luna-none`, `analysis.py`):** Simulates crowds of $N = 30$ rule-based people starting from a shared $1.5\times$ directional mistake who repeatedly query a single LLM assistant (`GPT-6 Luna`) over $T = 6$ rounds on post-cutoff 2026 events with verified ground truth. Each round, people move $30\%$ of the way toward the assistant's estimate ($x_{i,t+1} = 0.7 x_{i,t} + 0.3 a_{i,t}$), and crowd error is decomposed exactly via Page's Diversity Prediction Theorem.
2. **Track 4 Categorical & Multi-Backend Suite (`src/apart_epistemics/`):** Extends evaluation to multi-model provenance (`nationality_type`), model count ($K$), decoding temperature, and blinded dual-judge topic clustering, alongside deterministic synthetic demos for zero-cost offline exploration.

The frontend uses React, TypeScript, Vite, Tailwind CSS, and shadcn/ui components, running directly from saved report artifacts with no API keys required.

---

## Why It Exists

Crowds are wise when individual errors point in different directions and cancel out (Galton, 1907; Page, 2007). When many people consult the same AI assistant, their errors may stop canceling out:

- **Epistemic Herding Without Sycophancy:** Even when an assistant barely defers to individual users (moving only $11\text{–}14\%$ toward each person's estimate under low reasoning), repeated consultation causes everyone to hear the same voice, halving crowd spread while locking in the assistant's mistakes.
- **Deployment Information Regimes (`SOLO` / `PEER` / `TAILORED`):** Loosely adapted from Thejaswi et al. (2026), we test whether exposing the assistant to only the user's estimate (`SOLO`, C1), adding the crowd's median and interquartile spread (`PEER`, C2), or adding a user profile and conversation history (`TAILORED`, C3) alters convergence or accuracy.
- **Post-Cutoff Estimation vs. Recall:** Standard benchmarks conflate parametric memorization with reasoning. By pairing 2026 post-cutoff events (where cold model error exceeds $26\%$) against memorized pre-cutoff census controls, the testbed separates genuine error correction from shared echo.
- **Hidden Reasoning & Echo Susceptibility:** Toggling hidden reasoning off (`reasoning_effort=none`) dramatically increases user echoing (from $0.14$ to $0.61$ in `SOLO`) and reveals statistically significant diversity differences across deployment conditions.

---

## Key Empirical Highlights

Results from the full 17,820-query main run (`results/numeric-final-1114`, `GPT-6 Luna`, low reasoning effort) and reasoning-off ablation (`results/numeric-final-luna-none`):

- **Crowds Converged on the Assistant, Mistakes Included:** Across the eight main post-cutoff questions, crowds started $1.53\times$ off from the truth and ended $1.41\times$ (`SOLO`), $1.38\times$ (`PEER`), and $1.39\times$ (`TAILORED`) off after 6 rounds—sitting within a few percent of the assistant's own biased priors while typical individual spread around the crowd center collapsed from $1.72\times$ to $1.11\text{–}1.17\times$.
- **Prompt Context Did Not Reliably Improve Accuracy:** Adding crowd summaries (`PEER` vs. `SOLO`, $p_{\text{Holm}} = 0.95$) or user profiles and history (`TAILORED` vs. `PEER`, $p_{\text{Holm}} = 0.59$) produced no statistically reliable gain in final crowd error, while cutting diversity roughly in half relative to `SOLO`.
- **Controls Confirmed Correction When Facts Were Known:** On pre-cutoff Indian census controls (Nahan, Kullu) where the model knew the exact answer, crowds converged to within $1.05\times$ of truth across all conditions.
- **Hidden Reasoning Dampens User Echoing:** Switching hidden reasoning off increased the round-1 user echo slope in `SOLO` from $0.14$ to $0.61$ (reaching $1.00$ outright copying on box-office questions) and produced significant diversity shifts across conditions (`PEER < SOLO`, $p_{\text{Holm}} = 0.031$; `TAILORED > PEER`, $p_{\text{Holm}} = 0.047$) without improving accuracy.

| Condition | Round-6 Crowd Error ($\log_{10}^2$) | Crowd Center Offset | Round-6 Diversity ($\log_{10}^2$) | Typical Spread Offset | Round-1 User Echo Slope |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SOLO** (User estimate only) | `0.0222` | `1.41x` | `0.0048` | `1.17x` | `0.139` (`0.61` reasoning off) |
| **PEER** (+ Crowd median & IQR) | `0.0193` | `1.38x` | `0.0024` | `1.12x` | `0.107` (`0.17` reasoning off) |
| **TAILORED** (+ Profile & history) | `0.0206` | `1.39x` | `0.0021` | `1.11x` | `0.111` (`0.19` reasoning off) |

---

## Features

- **Dual-Track Simulation Engine (`src/apart_epistemics/engine/`):** Supports damped $\log_{10}$ quantitative belief updating (`SOLO`/`C1`, `PEER`/`C2`, `TAILORED`/`C3`) with Page (2007) error-diversity decomposition and exact Wilcoxon / Holm-Bonferroni contrast analysis (`analysis.py`, `contrasts.py`), alongside multi-round categorical debate simulations (`GROUND_TRUTH`, `OPEN_DEBATE`).
- **Post-Cutoff Benchmark & Calibration Suite (`configs/`):** Curated 2026 post-cutoff estimation questions (Asian Games, Commonwealth Games, US-Canada box office) split into high- and low-spread strata, paired with pre-cutoff census controls and a pre-registered movability check.
- **Protocol-Clean Multi-Backend Adapters (`src/apart_epistemics/backends/`):** OpenRouter (`openai/gpt-6-luna`, `allow_fallbacks: false`), Adaption (`gemma-4-26b-a4b-it`, `qwen3.5-397b`), OpenAI, Anthropic, Gemini, local Ollama, offline `ReplayBackend`, and deterministic `SyntheticEpistemicBackend`.
- **Crash-Resumable JSONL & Budget Guard:** Skips completed `(condition_id, question_id, replicate, round_idx, agent_id)` cells on restart, retries once on format errors, logs unparseable replies safely as `miss=True`, and enforces persistent USD request reservations via `--budget`.
- **Blinded Dual-Judge Topic Summarizer (`summarizer.py`):** Strips condition, model, and round metadata, hashes deterministic `blind_id`s, enforces Cohen's $\kappa \ge 0.70$, and exports stratified `audit_human_check.csv`.
- **Interactive React Explorer (`frontend/`):** Zero-external-dependency explorer with round scrubbers, condition/stratum filters, sortable ledgers, and full prompt-and-reply trace drawers.

---

## Architecture & Project Structure

- `src/apart_epistemics/` — Core Python library (`datasets/`, `backends/`, `engine/`, `report/`, `models.py`, `parsers.py`, `metrics.py`, `summarizer.py`, `cli.py`).
- `results/` — Saved run manifests and JSONL request/reply logs (`numeric-final-1114` main run, `numeric-final-luna-none` reasoning-off ablation, and pilot sweeps).
- `analysis.py` — Recomputes all Page (2007) decompositions, echo slopes, bootstrap intervals, and Holm-adjusted Wilcoxon tests directly from saved logs.
- `configs/` — Task definitions, prompt templates, rounding rules, and selection guards.
- `frontend/` — React + TypeScript + Vite web explorer deployed to GitHub Pages.
- `reports/` — Generated tables, figures, and model evaluation artifacts.
- `tests/` — Offline unit, integration, manifest-resume, budget-guard, and Playwright browser tests.

---

## Python Quick Start

See `HACKATHON.md` for the full onboarding checklist, Windows commands, and live-run safety caveats. No API keys or model downloads are required for offline analysis or the synthetic demo.

```bash
git clone [https://github.com/vikramwish/apart-epistemics.git](https://github.com/vikramwish/apart-epistemics.git)
cd apart-epistemics

make install     # Python 3.11+; no uv required
make check       # Lint, fast tests, and small synthetic demo in tmp/
make serve       # Open [http://127.0.0.1:8000](http://127.0.0.1:8000); Ctrl-C to stop

# Recompute paper statistics directly from the released final logs (zero API cost)
python analysis.py

# Optional: browser tests (requires Playwright Chromium)
make install-browser
make test

# Optional: rebuild full offline benchmark and tracked dashboard
make demo
