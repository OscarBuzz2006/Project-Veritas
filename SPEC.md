# Specification: Epistemic Prism (`apart-epistemics`)

> **One-sentence goal:** A reproducible multi-agent epistemic simulation harness and interactive benchmark explorer for measuring how monolithic, multi-model, culturally diverse, and personalized AI assistants alter **topic diversity**, **source independence**, **accuracy**, and **confidence calibration** across multi-round human–AI belief updating.

---

## 1. Problem & Research Gap

Recent empirical studies establish that LLMs compress stylistic diversity (*Shrinking Landscape*, Sourati et al. 2025a), flatten long-form arguments across frontier labs (*Argument Collapse*, Kim et al. 2026), and exhibit correlated errors across 350+ models (Kim et al. 2025). Yet three critical gaps remain for the Apart Research AI & Epistemics study:

1. **Decoding artifacts vs. epistemic collapse:** Prior rewrite experiments often ran at `temperature=0, top_p=0`, conflating greedy decoding determinism with deeper mode collapse. We need to isolate *surface lexical variance* from *epistemic stance & topic collapse* across greedy (`T=0, top_p=0`), app-default (`T=0.8, top_p=0.95`), and high-entropy (`T=1.3`) decoding.
2. **Cosine similarity misses epistemic failure:** Embedding distance correlates poorly with textual form (Qiu et al. 2026) and fails to detect when populations converge on being *confidently wrong* or lose source independence (Goldman 2001; DeMarzo et al. 2003).
3. **The personalization vs. calibration trade-off:** Mathematical models predict personalized assistants preserve diversity (Thejaswi et al. 2026), while empirical sycophancy studies warn personalization degrades epistemic independence when framed as a peer (Kelley & Riedl 2026). No multi-agent LLM harness jointly measures **topic diversity** and **ground-truth calibration** across model count ($K$), heterogeneity type, and role framing.

---

## 2. Users & Core Job

- **Primary users:** Apart Research hackathon researchers (`vikram.vish`, `Oscar`, `dewlegz`, `Kevin O'Shaughnessy`, `musif-80`) and AI epistemics / safety researchers.
- **Core job:** Configure and run multi-round populations of $N=20\text{–}50$ persona-grounded agents across experimental arms, log structured belief/topic/source traces, compute calibration and Goldman source-independence metrics, and inspect results visually in an interactive report.

### Non-Goals (Explicitly Out of Scope for v1)
1. **Not a training/fine-tuning framework:** We evaluate inference-time and multi-agent dynamics without RLHF/SFT weight training.
2. **Not a production chat UI or user-auth SaaS app:** Results are explored via static self-contained HTML reports (`docs/index.html`) and CLI/JSONL exports.
3. **Not a paid-API-only dependency:** Every simulation, test, SVG plot, and dashboard report must run deterministically offline (`synthetic` and `replay` backends) as well as against local `ollama` (`llama3.1:8b`) and optional env-configured cloud APIs.

---

## 3. User Stories

1. **Researcher running the full benchmark:** As a researcher, I can run `apart-epistemics demo` to execute all experimental arms (`no_ai`, `monolithic_shared`, `multi_lab_mix`, `cultural_regional_mix`, `personalized_advisor`, `personalized_peer`), model-count sweeps ($K \in \{1, 2, 4\}$), and decoding ablations (`T=0, top_p=0` vs. `T=0.8, top_p=0.95`), outputting JSONL traces, summary CSV/JSON, SVG figures, and an interactive HTML explorer.
2. **Researcher auditing the `T=0` decoding confound:** As a researcher, I can compare `monolithic_shared` under greedy (`T=0, top_p=0`) vs. app-default (`T=0.8, top_p=0.95`) decoding side-by-side to quantify how much surface lexical variance changes vs. deep topic/belief entropy.
3. **Researcher testing local open-weight models for free:** As a researcher with Ollama running locally (`llama3.1:8b`), I can run `apart-epistemics run --backend ollama` and produce the exact same `RoundRecord` JSONL schema and metrics.
4. **Researcher plugging in multi-lab API keys:** As a researcher with keys in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`), I can run live multi-provider mixes without modifying Python code.
5. **Reviewer / teammate exploring results:** As a teammate or hackathon judge opening the deployed GitHub Pages link (`docs/index.html`), I can filter by task track, scrub rounds $0 \rightarrow 5$, inspect the calibration reliability diagram and double-dissociation chart, and drill into individual round traces on desktop (1440px) and mobile (390px).

---

## 4. Data Model & Experimental Schema

- **`TaskItem`**: `id`, `kind` (`ground_truth` | `open_debate`), `domain`, `question`, `hypothesis_a`, `hypothesis_b`, `ground_truth` (`0 | 1 | None`), `popular_misconception_bias` (`float`), `global_north_prior` (`float`), `canonical_topics` (`list[str]`).
- **`AgentProfile`**: `agent_id`, `name`, `region`, `prior_belief` ($p_i^{(0)} \in [0.05, 0.95]$), `prior_confidence` ($c_i^{(0)} \in [0.5, 0.99]$), `need_for_cognition` ($\in [0, 1]$), `preferred_topics` (`list[str]`).
- **`ConditionConfig`**:
  - `condition_id`, `label`, `heterogeneity_type` (`no_ai`, `monolithic_shared`, `multi_lab_mix`, `cultural_regional_mix`, `personalized_advisor`, `personalized_peer`)
  - `num_models` ($K \ge 0$), `temperature` ($\in [0.0, 2.0]$), `top_p` ($\in [0.0, 1.0]$)
  - `num_agents` (default `24`, configurable `20–50`), `num_rounds` (default `5`), `peer_degree` (default `3`), `seed`
- **`RoundRecord`**: `run_id`, `condition_id`, `seed`, `task_id`, `task_kind`, `round_idx`, `agent_id`, `agent_region`, `model_assigned`, `model_family`, `assistant_response_text`, `agent_response_text`, `extracted_topics`, `stance_prob`, `confidence`, `stated_answer`, `goldman_source_overlap`, `persuasion_bias_exposure`.

---

## 5. Metrics & Mathematical Definitions

1. **Calibration & Truth-Tracking (`ground_truth` tasks):**
   - `accuracy`: fraction where `(stance_prob >= 0.5) == ground_truth`
   - `brier_score`: $\frac{1}{N}\sum_{i=1}^N (p_i - y)^2$
   - `ece`: 10-bin Expected Calibration Error $\sum_{b=1}^{10} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$
   - `confident_wrong_rate`: fraction with wrong answer and `confidence >= 0.75`
2. **Goldman Source Independence & DeMarzo Drift:**
   - `effective_independent_sources` ($N_{\text{eff}}$): Kish / covariance-adjusted effective sample size $\frac{N^2}{\sum_{i,j} \hat{\rho}_{ij}}$, where $\hat{\rho}_{ij}$ measures shared underlying model/family provenance and peer echo overlap.
   - `persuasion_bias_drift`: mean excess posterior shift driven by repeated correlated model/peer echoes relative to Round 0 independent priors.
3. **Topic & Epistemic Diversity (`open_debate` tasks):**
   - `topic_shannon_entropy`: normalized Shannon entropy $H / \log_2 |\mathcal{V}|$ over canonical topic occurrences across agents.
   - `effective_topic_count`: Hill number $\exp(H_{\text{nat}})$ measuring effective distinct conceptual angles surviving in the population.
   - `stance_std` & `bimodality_coefficient`: separates smooth pluralism (`high topic entropy`, `moderate BC < 0.555`) from echo-chamber polarization (`BC > 0.555`).
   - `global_north_skew_shift`: absolute reduction in distance toward the Global North / WEIRD modal prior across rounds.
4. **Surface Lexical Diversity (contrasted with epistemic diversity):**
   - `distinct_2`: unique bigrams / total bigrams in agent texts.
   - `mean_pairwise_jaccard_distance`: mean $1 - |A \cap B| / |A \cup B|$ across pairs of agent outputs on the same task and round.

---

## 6. Testable Acceptance Criteria

- **AC1 (CLI & Demo Suite):** `apart-epistemics demo` completes cleanly in `< 15s`, creating `runs.jsonl`, `summary.json`, `summary.csv`, 4 SVG charts, and `docs/index.html`.
- **AC2 (Decoding Double Dissociation):** Comparing `monolithic_shared` at `(T=0.0, top_p=0.0)` vs. `(T=0.8, top_p=0.95)` shows a statistically clear jump in `distinct_2` / `mean_pairwise_jaccard_distance` ($\Delta > 0.08$) while `topic_shannon_entropy` and `effective_independent_sources` remain severely collapsed relative to `no_ai` and `cultural_regional_mix`.
- **AC3 (Personalization Trade-off):** `personalized_peer` preserves higher `effective_topic_count` and `stance_std` than `monolithic_shared`, yet exhibits higher `confident_wrong_rate` and `brier_score` than `personalized_advisor`.
- **AC4 (Model Count vs. Type IVs):** Holding `type` fixed and increasing $K$ (`1 -> 2 -> 4`) vs. changing `type` (`multi_lab_mix` vs. `cultural_regional_mix`) produces distinct measurable shifts in `effective_independent_sources` and `topic_shannon_entropy`.
- **AC5 (Backend Parity & Error Safety):** `replay` reproduces identical metrics from recorded JSONL; `ollama` client cleanly serializes `temperature` / `top_p` and handles unreachable server errors cleanly; live API adapters fail cleanly with helpful instructions when env keys are missing and never log secrets.
- **AC6 (Dashboard Visual & Responsive Quality):** Playwright renders `docs/index.html` cleanly at `1440x900` and `390x844` with zero console errors, zero horizontal overflow, working condition/round/track filters, and passing accessibility checks.

---

## 7. Risks & Assumptions

- **Assumption 1 (Synthetic Calibration Parameters):** The `synthetic` backend encodes directional behavioral dynamics grounded in cited literature (Kim et al. 2025/2026, Kelley & Riedl 2026, Tao et al. 2024, DeMarzo et al. 2003); empirical magnitudes for real API runs depend on live provider queries and are explicitly labeled `synthetic` in reports and figures.
- **Assumption 2 (No Paid API Keys in CI):** CI runs all unit, integration, and Playwright E2E tests against `synthetic` and `replay` backends; live `ollama` / cloud API tests are guarded or mocked in CI.

