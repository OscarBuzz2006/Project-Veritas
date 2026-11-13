# Codebase audit — 14 November 2026

Scope: Python simulation engines, provider adapters, parsers, metrics/statistics,
datasets, CLI, report generation, React data flow and desktop views, tests, and CI.
Also inspected the additional `propaganda-eval` scripts in the local Oscar sync
checkout. This is a source review with targeted offline reproductions, not a
certification of correctness or a live-provider integration test.

**Resolution:** the fixes below are implemented in the accompanying changes.
Regression tests cover routing, parsing, validated resume, budget persistence,
trace preservation, redaction, partial-run analysis, and unsupported replay.
Default judge reliability is now unavailable; proxy metrics are labeled.
Oscar-specific evaluator fixes are applied separately in his checkout.
Validation after fixes: 60 Python tests, 20 React tests, and 5 desktop browser
checks, plus 4 additional evaluator regressions in Oscar’s repository.

The following describes the original findings before these changes.

The highest-risk findings affect experiment validity and saved results. Passing
UI and unit tests currently does not establish that live experiments implement
their configured conditions correctly. Application behavior was subsequently corrected as described above.

## High priority

### 1. Configured model diversity does not route to different live models

Sources: `src/apart_epistemics/engine/track4.py:115`,
`src/apart_epistemics/backends/providers.py:52`.

Track 4 puts `model_assigned` into a system prompt, but calls a single backend
whose request payload always uses its fixed `self.model`. Records retain the
configured assignment. A two-model offline reproduction sent every request to
`actual-fixed-model`. Model-count and nationality comparisons therefore do not
test the stated model pools in live mode.

Fix: route assistant calls through adapters for the actual assigned model and
persist the returned provider/model identity separately from intended assignment.
Keep the agent simulator's model explicit too. Test outgoing request payloads.

### 2. The response parser rejects the format the live prompt requests

Sources: `src/apart_epistemics/engine/track4.py:135`,
`src/apart_epistemics/parsers.py:99`.

The prompt requests a JSON object followed by two sentences. If the response
starts with `{`, the parser passes the entire response to `json.loads`, which
rejects trailing prose. A valid object containing stance 0.8 followed by prose
returned NaN stance, confidence 0.5, and no topics. The engine then preserves the
prior stance and previous topics, silently making updates look absent.

Fix: decode the leading JSON object with a proper decoder, validate its fields,
and record parse status. Test the exact requested response format.

### 3. Numeric resume silently reuses records from incompatible experiments

Source: `src/apart_epistemics/engine/numeric_core.py:198`.

The resume key contains condition, question ID, replicate, round, and agent ID.
It does not validate seed, model/backend, alpha, population size, or question
content. Changing seed 42 to 99, alpha 0.3 to 1, and backend reused all four
existing rows without calling the new backend. Partial files can mix runs.

Fix: persist and validate a run manifest/fingerprint before resuming. Refuse
incompatible reuse and require a new output directory or explicit new run.

### 4. The budget does not constrain the whole run

Sources: `src/apart_epistemics/cli.py:237`,
`src/apart_epistemics/engine/numeric_core.py:168`.

Track 4 runs before the budget is checked by the numeric engine. A minimal
two-track run with budget zero made 24 fake provider calls before stopping.
Numeric spending also resets on resume, uses fixed assumed prices, and checks
the cap before a call without reserving its possible cost.

Fix: use one persisted budget ledger across both tracks and retries, reject a
zero budget before any paid request, and distinguish estimated cost from actual
provider billing. Use model-specific pricing or clearly bounded reservations.

### 5. A failed Track 4 rerun destroys existing traces

Source: `src/apart_epistemics/engine/track4.py:493`.

`runs.jsonl` is opened with `w` before any provider calls. Each condition is
computed in memory before its records are written. A failure on the first call
reduced an existing 16,061-byte trace file to zero bytes. A mid-condition failure
also loses successful responses already paid for in that condition.

Fix: protect existing output, use run-specific paths or explicit validated
resume, and persist each completed observation as it becomes available.

### 6. Default topic reliability is circular

Source: `src/apart_epistemics/summarizer.py:91`.

Both default judges are overwritten with the same seed topics. The secondary
keyword judge always returns a nonempty fallback, so the supposed grounding
check cannot reject seed topics. Replacing every response with unrelated text
while retaining alternating topic labels still produced kappa 1.0. This path is
also used for live runs; it is not independent verification of their topics.

Fix: distinguish seeded synthetic labels from independent judging. Evaluate
judges from blinded text without copying target labels, and mark reliability
unavailable when independent judgments have not been collected.

### 7. Redacted errors retain secrets in chained tracebacks

Source: `src/apart_epistemics/backends/base.py:101`.

The outer exception message is sanitized but is raised `from exc`, retaining
the original message. An injected fake secret disappeared from `str(error)`
but remained in `traceback.format_exc()`. Gemini places its key in the request
URL, which transport errors can include. This affects logged full tracebacks;
the CLI's sanitized short error path alone does not expose the same text.

Fix: prevent unsanitized exception chains from reaching logs and remove secrets
from URLs/messages at the transport boundary. Test complete formatted tracebacks.

## Medium priority

### 8. Partial numeric runs can produce misleading final results

Sources: `src/apart_epistemics/engine/numeric_core.py:312`,
`src/apart_epistemics/engine/contrasts.py:30`.

The analyzer uses one global maximum round and does not validate cell
completeness. With C1 through round 2 and C2 through round 1, only C1 appeared in
final results. The missing C2-vs-C1 contrast still had effect 0, CI [0, 0], p=1,
and n_cells=0. The UI renders these numbers as recorded contrasts. An interrupted
cell can also be analyzed from only a subset of agents.

Fix: carry expected dimensions and completion status into analysis; show missing
comparisons as unavailable and identify incomplete cells explicitly.

### 9. Live source-diversity and persuasion metrics include fixed assumptions

Sources: `src/apart_epistemics/engine/track4.py:167`,
`src/apart_epistemics/metrics.py:195`.

Live source overlap is assigned 0.6 or 0.25 from configured model count;
persuasion exposure is always 0.5. Effective-source correlations use fixed
constants based on assigned model/family labels. These can encode the expected
ordering rather than measure independence in actual responses, especially with
the routing problem above.

Fix: label these as assumption-based proxies everywhere, or replace them with
validated measurements. Preserve the assumptions and actual backend provenance
in exported reports. The current report metadata does not identify the backend.

### 10. Numeric replay is accepted but unsupported

Sources: `src/apart_epistemics/backends/replay.py`,
`src/apart_epistemics/cli.py:444`.

The CLI accepts replay for paths that invoke the numeric engine, but ReplayBackend
only exposes Track 4 records. A fresh numeric run fails with
`AttributeError: 'ReplayBackend' object has no attribute 'generate_numeric_reply'`.

Fix: implement numeric replay or reject unsupported backend/track combinations
before execution and before output files are modified.

### 11. Oscar's additional evaluation scripts have broken default dataset paths

Sources in the local Oscar checkout: `propaganda-eval/grade.py:63` and
`propaganda-eval/score.py:18`.

Both default to `data/items.jsonl` in the repository root. That path is absent;
the dataset is beside the scripts at `propaganda-eval/items.jsonl`. Default
invocations therefore fail to open the dataset unless `--items` is supplied.

Fix: resolve the sibling dataset and test default CLI invocations. This finding
applies to the Oscar-specific files, not the current main checkout.

## Additional review notes

- H4 pools assistant and updated-agent estimates across questions of different
  magnitudes. The update itself includes the assistant estimate. This correlation
  is descriptive and mechanically coupled; it should not be presented as an
  independent causal measure of model influence.
- `backends/synthetic.py:514` breaks tied topic counts using set iteration order,
  which can vary across Python processes despite a fixed random seed.
- The audit sampler takes the first shuffled records despite describing itself
  as stratified. It does not guarantee representation from each condition.
- Prompts, tool events, and skill events are not persisted in current trace
  records. The frontend correctly marks them as not recorded; UI work alone
  cannot recover this missing provenance.
- The frontend embeds the exported JSON at build time. New simulation results
  require export and rebuild; this is a static report, not a live run monitor.
- The desktop CSS has accumulated repeated overrides. Consolidation would reduce
  regression risk, but it is lower priority than the correctness issues above.

## Validation and limits

- Python: all 44 existing tests passed, including the browser test after allowing
  its local server to run outside the filesystem/network sandbox.
- Frontend: 17 unit tests and all 5 desktop Playwright tests passed.
- Ruff lint/format, frontend TypeScript/format checks, and production build passed.
  The build emitted nonfatal third-party `use client`/source-map warnings.
- Offline reproductions confirmed findings 1–8 and 10; findings 9 and 11 were
  verified directly against source and filesystem paths. No paid API calls ran.
- Codebase-memory coverage reported no recorded issues for the cited core Python
  paths. CSS had a partial-parse flag, so graph coverage was not treated as proof
  of complete source coverage.
- No new live-provider integration, dependency vulnerability, or penetration
  testing was performed. Passing tests do not cover the reproduced failures.

Recommended order: fix routing and parsing; protect traces and validate resume;
enforce budget and secret handling; then correct reliability, incomplete-run
analysis, and unsupported CLI combinations. Add regression tests for each exact
failure before using live runs as evidence.
