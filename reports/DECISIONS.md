# Architectural & Research Decisions Log

## 2026-11-13 — Initial Architecture & Design Decisions

1. **Repo Name & Branding (`Epistemic Prism` / `apart-epistemics`)**
   - **Decision:** Name the toolkit and dashboard **Epistemic Prism** under repository `apart-epistemics`.
   - **Why:** Signals the core measurement contribution—splitting apparently uniform AI discourse into distinct dimensions (surface phrasing vs. conceptual topic breadth vs. calibration vs. source independence).
   - **Alternatives rejected:** Generic names like `llm-diversity-sim` (too narrow; ignores calibration and Goldman source independence).

2. **Two-Axis IV + Dual-Track DV Experimental Design**
   - **Decision:** Separate **Model Count ($K$)** from **Heterogeneity Type** (`no_ai`, `monolithic_shared`, `multi_lab_mix`, `cultural_regional_mix`, `personalized_advisor`, `personalized_peer`), add **Decoding Regime** (`T=0.0, top_p=0.0` vs. `T=0.8, top_p=0.95`) as a targeted ablation, and evaluate on **Ground-Truth Calibration Tasks** + **Open-Ended Debate Tasks**.
   - **Why:** Reconciles Oscar's voice-note IVs and topic-diversity DV with `dewlegz` / `musif-80`'s calibration & Goldman source-independence requirements and `dewlegz`'s `temperature=0, top_p=0` finding.
   - **Alternatives rejected:** Testing only cosine embedding similarity (already refuted by Qiu et al. 2026 and Oscar) or only 1-model vs. 5-US-labs (overlaps with Kim et al. 2026 *Argument Collapse*).

3. **Backend Architecture (`synthetic`, `ollama`, `live_http`, `replay`)**
   - **Decision:** Build a protocol-clean multi-backend abstraction with a deterministic, mathematically grounded `synthetic` backend as default for tests/CI/offline demos, plus `ollama` (tested with local `llama3.1:8b`), stdlib HTTP cloud adapters, and JSONL `replay`.
   - **Why:** Allows full end-to-end execution, TDD, CI, and interactive visualization without requiring paid cloud API keys, while remaining drop-in ready for multi-provider live API runs when keys are supplied in `.env`.
   - **Alternatives rejected:** Hard-coding `openai` SDK calls everywhere (impossible to test offline or run cleanly in GitHub Actions without secrets).

4. **Zero-CDN Self-Contained Interactive Dashboard + SVG Figures**
   - **Decision:** Generate `docs/index.html` as a self-contained static web explorer using inline SVGs, local/system editorial typography stacks, and vanilla JS, alongside standalone SVG charts in `reports/figures/`.
   - **Why:** Deployable directly to GitHub Pages with zero build-step fragility, instant load speed, offline reproducibility, and reliable Playwright verification at `1440px` and `390px`.
   - **Alternatives rejected:** Heavy React/Next frontend or external CDN chart libraries (breaks offline snapshots and adds needless Node build complexity to a Python research package).

5. **License Choice**
   - **Decision:** MIT License.
   - **Why:** Standard permissive open-source license for research benchmarks and portfolio repositories.

## 2026-11-13 — Implementation & Verification Refinements

6. **Per-Task Normalized Topic Entropy & Hill $N_1$ Aggregation**
   - **Decision:** Compute `compute_topic_entropy_and_hill` per task against that task's canonical codebook, then average across tasks per condition.
   - **Why:** Makes normalized Shannon entropy and Hill effective topic counts (`[1.0, ~6.5]`) invariant to `tasks_per_track` rather than conflating cross-question vocabulary growth with genuine multi-perspective depth.
   - **Alternatives rejected:** Pooling raw topic tokens across all tasks into one global histogram (artificially inflates $N_1$ proportionally to prompt count).

7. **Git-Ignore Large Raw JSONL Traces While Tracking Compact Summaries & Explorer Artifacts**
   - **Decision:** Ignore `reports/results/*.jsonl` (~20 MB generated on each deterministic `demo` invocation) in `.gitignore`, while committing `summary.csv`, `summary.json`, `numeric_analysis.json`, `audit_human_check.csv`, `selection_audit.json`, `reports/figures/*.svg`, and `docs/`.
   - **Why:** Keeps repository clone size lightweight and diffs readable; `apart-epistemics demo` deterministically regenerates `runs.jsonl` and `numeric_runs.jsonl` in ~5 seconds.
   - **Alternatives rejected:** Committing 20 MB of deterministic synthetic JSONL lines into Git history.

8. **Playwright Visual Collision & Table-Width Regression Gates**
   - **Decision:** Added deterministic per-condition quadrant label offsets in both `report.py` and `dashboard.html`, widened SVG right-margin padding for `1.00` labels, wrapped `.cond-label-cell` cleanly, and added automated bounding-box assertions in `tests/e2e/test_dashboard_playwright.py` for unclipped `Inspect` buttons at `1440px` and zero horizontal scroll overflow at `390px`.
   - **Why:** First visual screenshot inspection revealed overlapping top-right quadrant labels and a slightly clipped table button at 1440px despite DOM assertions passing.
   - **Alternatives rejected:** Relying solely on HTML existence checks without rendering pixels.


