import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import benchmark from "../../docs/data/benchmark_summary.json";
import App from "./App";
import type { Benchmark } from "./lib/results";

describe("results explorer", () => {
  it("shows recorded counts and an explicit evidence caveat", () => {
    render(<App data={benchmark} />);
    expect(
      screen.getByRole("heading", { name: "Results explorer" }),
    ).toBeTruthy();
    expect(
      screen.getByText(
        benchmark.metadata.n_track4_records.toLocaleString("en-US"),
      ),
    ).toBeTruthy();
    expect(
      screen.getByText(/Synthetic runs demonstrate simulator assumptions/),
    ).toBeTruthy();
  });
  it("filters conditions and shows the selected round's recorded scores", () => {
    render(<App data={benchmark} />);
    fireEvent.change(
      screen.getByRole("combobox", { name: "Condition family" }),
      { target: { value: "baseline" } },
    );
    const table = screen.getByRole("table", { name: "Condition scores" });
    expect(within(table).getAllByRole("row")).toHaveLength(2);
    fireEvent.change(
      screen.getByRole("combobox", { name: "Communication round" }),
      { target: { value: "0" } },
    );
    expect(
      within(table).getByText(
        benchmark.summaries[0].round_trajectory[0].brier_score.toFixed(3),
      ),
    ).toBeTruthy();
    fireEvent.change(
      screen.getByRole("searchbox", { name: "Search conditions" }),
      { target: { value: "no-such-condition" } },
    );
    expect(screen.getByText("No conditions match these filters.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset filters" }));
    expect(within(table).getAllByRole("row")).toHaveLength(
      benchmark.summaries.length + 1,
    );
  });
  it("opens genuine response messages with expandable recorded context", () => {
    render(<App data={benchmark} />);
    fireEvent.click(screen.getByRole("button", { name: "Inspect responses" }));
    const specimen = benchmark.trace_specimens[0];
    expect(
      screen.getByRole("heading", { name: "Recorded response specimens" }),
    ).toBeTruthy();
    expect(
      screen.getAllByText(specimen.assistant_response_text).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText(specimen.agent_response_text)).toBeTruthy();
    const context = screen.getByRole("button", {
      name: "Recorded context · specimen 1",
    });
    expect(context.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(context);
    expect(context.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText(specimen.model_family)).toBeTruthy();
    expect(screen.queryByText("Thinking…")).toBeNull();
  });
  it("shows the numeric decomposition and original statistical contrasts", () => {
    render(<App data={benchmark} />);
    fireEvent.keyDown(screen.getByRole("tab", { name: "Numeric core" }), {
      key: "Enter",
    });
    expect(
      screen.getByRole("heading", { name: "Page's diversity identity" }),
    ).toBeTruthy();
    const table = screen.getByRole("table", {
      name: "Numeric final-round scores",
    });
    expect(within(table).getAllByRole("row")).toHaveLength(
      Object.keys(benchmark.numeric_analysis.by_condition_final).length + 1,
    );
    expect(
      within(table).getByText(
        benchmark.numeric_analysis.by_condition_final.C3.crowd_error.toFixed(6),
      ),
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Shared population median and IQR + persona and conversation history",
      ),
    ).toBeTruthy();
    expect(
      within(
        screen.getByRole("table", { name: "Numeric contrasts" }),
      ).getAllByRole("row"),
    ).toHaveLength(benchmark.numeric_analysis.contrasts.length + 1);
  });
  it("sorts scores without changing the source data", () => {
    const source = JSON.stringify(benchmark);
    render(<App data={benchmark} />);
    const table = screen.getByRole("table", { name: "Condition scores" });
    fireEvent.click(within(table).getByRole("button", { name: "Brier score" }));
    const best = [...benchmark.summaries].sort(
      (a, b) => a.brier_score - b.brier_score,
    )[0];
    expect(within(table).getAllByRole("row")[1].textContent).toContain(
      best.condition_id,
    );
    expect(
      screen
        .getByRole("columnheader", { name: "Brier score" })
        .getAttribute("aria-sort"),
    ).toBe("ascending");
    expect(JSON.stringify(benchmark)).toBe(source);
  });
  it("allows keyboard chart selection and changes the trajectory metric", () => {
    render(<App data={benchmark} />);
    const selected = benchmark.summaries[2];
    fireEvent.keyDown(
      screen.getByRole("button", { name: `Select ${selected.label}` }),
      { key: "Enter" },
    );
    expect(screen.getByRole("heading", { name: selected.label })).toBeTruthy();
    fireEvent.change(
      screen.getByRole("combobox", { name: "Trajectory metric" }),
      { target: { value: "brier_score" } },
    );
    expect(
      screen.getByRole("img", {
        name: `${selected.label}: Brier score by round`,
      }),
    ).toBeTruthy();
  });
  it("renders stored response markup as literal text, never executable HTML", () => {
    const data = structuredClone(benchmark);
    const text = '<img src=x onerror="alert(1)" data-untrusted="yes">';
    data.trace_specimens[0].agent_response_text = text;
    const { container } = render(<App data={data} />);
    fireEvent.click(screen.getByRole("button", { name: "Inspect responses" }));
    expect(screen.getByText(text)).toBeTruthy();
    expect(container.querySelector("[data-untrusted]")).toBeNull();
  });
  it("shows empty states rather than fabricated scores or traces", () => {
    const data: Benchmark = {
      ...benchmark,
      summaries: [],
      trace_specimens: [],
      numeric_analysis: {
        ...benchmark.numeric_analysis,
        by_condition_final: {},
        contrasts: [],
        condition_round_metrics: [],
      },
    };
    render(<App data={data} />);
    expect(screen.getByText("No conditions match these filters.")).toBeTruthy();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Response traces" }), {
      key: "Enter",
    });
    expect(
      screen.getByText(
        "No response specimens are available for this selection.",
      ),
    ).toBeTruthy();
    fireEvent.keyDown(screen.getByRole("tab", { name: "Numeric core" }), {
      key: "Enter",
    });
    expect(screen.getByText("No numeric results are available.")).toBeTruthy();
    expect(screen.queryByText(/NaN|Infinity/)).toBeNull();
  });
  it("does not silently substitute a final score for a missing round", () => {
    const data = structuredClone(benchmark);
    data.summaries[0].round_trajectory = [];
    render(<App data={data} />);
    fireEvent.change(
      screen.getByRole("combobox", { name: "Communication round" }),
      { target: { value: "0" } },
    );
    const row = screen
      .getByRole("table", { name: "Condition scores" })
      .querySelector('tr[data-selected="true"]')!;
    expect(within(row as HTMLElement).getAllByText("—")).toHaveLength(6);
    expect(
      screen.getByText("No recorded rounds for this selection."),
    ).toBeTruthy();
  });
});
