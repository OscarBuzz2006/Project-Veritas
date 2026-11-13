import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import benchmark from "../../docs/data/benchmark_summary.json";
import { NumericPanel } from "./components/numeric-panel";
import { ResponseInspector } from "./components/response-inspector";
import App from "./App";
import type { Benchmark } from "./lib/results";

it("renders unavailable judge reliability without converting it to zero", () => {
  render(<App data={benchmark} />);
  expect(
    screen.getByText(/No independent topic-label reliability/),
  ).toBeTruthy();
  expect(screen.getByText(/Judge agreement κ = Unavailable/)).toBeTruthy();
});

it("explains withheld final comparisons on an interrupted numeric run", () => {
  const numeric: Benchmark["numeric_analysis"] = {
    ...benchmark.numeric_analysis,
    completion_status: "incomplete_or_unverified",
    by_condition_final: {},
    contrasts: [],
  };
  render(<NumericPanel numeric={numeric} />);
  expect(screen.getByRole("status").textContent).toContain(
    "Final comparisons are unavailable",
  );
  expect(
    screen.getByText("No statistical contrasts were recorded."),
  ).toBeTruthy();
});

it("displays saved live system prompts and actual routing provenance", () => {
  const selected = benchmark.summaries.find((s) => s.num_models === 1)!;
  const source = benchmark.trace_specimens.find(
    (s) => s.condition_id === selected.condition_id,
  )!;
  render(
    <ResponseInspector
      conditions={[selected]}
      selected={selected}
      specimens={[
        {
          ...source,
          provenance: {
            assistant_system_prompt: "Saved assistant instructions",
            agent_system_prompt: "Saved agent instructions",
            assistant_model_actual: "provider/actual-model",
            parse_status: "parsed",
          },
        },
      ]}
      onSelect={() => {}}
      baselineConditions={[]}
      onSelectBaseline={() => {}}
    />,
  );
  fireEvent.click(
    screen.getByRole("button", { name: "System prompts · specimen 1" }),
  );
  expect(screen.getByText("Saved assistant instructions")).toBeTruthy();
  expect(screen.getByText("Saved agent instructions")).toBeTruthy();
  fireEvent.click(
    screen.getByRole("button", { name: "Recorded context · specimen 1" }),
  );
  expect(screen.getByText("provider/actual-model")).toBeTruthy();
});
