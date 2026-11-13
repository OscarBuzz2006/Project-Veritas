import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it } from "vitest";
import benchmark from "../../docs/data/benchmark_summary.json";
import App from "./App";
import { readFileSync } from "node:fs";

const theme = readFileSync("src/index.css", "utf8");

it("defines theme animations used by the real trace accordion", () => {
  const { container } = render(<App data={benchmark} />);
  fireEvent.click(screen.getByRole("button", { name: "Inspect responses" }));
  const content = container.querySelector(
    '[data-slot="trace-collapsible-content"]',
  )!;
  for (const direction of ["down", "up"]) {
    expect(content.className).toContain(`animate-accordion-${direction}`);
    expect(theme).toContain(
      `--animate-accordion-${direction}: accordion-${direction}`,
    );
    expect(theme).toContain(`@keyframes accordion-${direction}`);
  }
  expect(theme).toContain("height: var(--radix-accordion-content-height)");
});

it("keeps chart numbers aligned with the condition list across missing rounds", () => {
  const data = structuredClone(benchmark);
  data.summaries[0].round_trajectory = [];
  data.summaries[2].round_trajectory = [];
  render(<App data={data} />);
  fireEvent.change(screen.getByLabelText("Communication round"), {
    target: { value: "0" },
  });
  const chart = screen.getByRole("region", {
    name: "Diversity and calibration chart",
  });
  const numbers = [...chart.querySelectorAll(".point-label")].map(
    (node) => node.textContent,
  );
  const expected = data.summaries.flatMap((condition, index) =>
    condition.round_trajectory.some((point) => point.round_idx === 0)
      ? [String(index + 1)]
      : [],
  );
  expect(numbers).toEqual(expected);
  const options = document.querySelectorAll(".condition-option");
  for (const number of numbers) {
    expect(
      options[Number(number) - 1].querySelector(".condition-number")
        ?.textContent,
    ).toBe(number);
  }
});

it("composes response paragraphs inside the real padded BubbleContent slots", () => {
  const { container } = render(<App data={benchmark} />);
  fireEvent.click(screen.getByRole("button", { name: "Inspect responses" }));
  const bubbles = container.querySelectorAll('[data-slot="bubble"]');
  expect(bubbles.length).toBeGreaterThan(0);
  for (const bubble of bubbles) {
    const content = bubble.querySelector(
      ':scope > [data-slot="bubble-content"]',
    );
    expect(content).not.toBeNull();
    expect(content!.classList.contains("px-3")).toBe(true);
    expect(content!.classList.contains("py-2")).toBe(true);
    expect(content!.querySelector("p.recorded-message")).not.toBeNull();
  }
  expect(bubbles[0].getAttribute("data-variant")).toBe("muted");
  expect(bubbles[1].getAttribute("data-variant")).toBe("outline");
});

it("wraps the actual personalized model badge within its assistant header", () => {
  const specimen = benchmark.trace_specimens.find(
    (record) =>
      record.model_assigned === "anthropic/claude-frontier@persona-agent_01",
  )!;
  expect(specimen).toBeDefined();
  render(<App data={benchmark} />);
  fireEvent.click(screen.getByRole("button", { name: "Inspect responses" }));
  fireEvent.change(screen.getByLabelText("Trace condition"), {
    target: { value: specimen.condition_id },
  });
  const article = screen.getByRole("article", {
    name: `Specimen 2: ${specimen.agent_id}`,
  });
  const badge = within(article).getByText(specimen.model_assigned);
  const header = badge.closest('[data-slot="message-header"]')!;
  expect(header.classList.contains("flex-wrap")).toBe(true);
  for (const className of [
    "max-w-full",
    "min-w-0",
    "whitespace-normal",
    "wrap-anywhere",
  ]) {
    expect(badge.classList.contains(className)).toBe(true);
  }
});
