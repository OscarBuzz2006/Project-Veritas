import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import benchmark from "../../docs/data/benchmark_summary.json";
import App from "./App";
import { colorFor } from "./lib/results";

it("keeps the report heading and interpretation note without decorative slogans", () => {
  const { container } = render(<App data={benchmark} />);
  expect(
    screen.getByRole("heading", { name: "Results explorer" }),
  ).toBeTruthy();
  expect(
    screen.getByText(/Synthetic runs demonstrate simulator assumptions/),
  ).toBeTruthy();
  expect(screen.queryByText("A lens, not a leaderboard.")).toBeNull();
  expect(
    container.querySelector(".title-dot, .dataset-stamp, .brand-mark"),
  ).toBeNull();
});

it("maintains readable white numbering on every condition color", () => {
  for (const condition of benchmark.summaries) {
    const color = colorFor(condition);
    const linear = [1, 3, 5].map((offset) => {
      const value = Number.parseInt(color.slice(offset, offset + 2), 16) / 255;
      return value <= 0.04045
        ? value / 12.92
        : ((value + 0.055) / 1.055) ** 2.4;
    });
    const luminance =
      linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
    expect(1.05 / (luminance + 0.05), condition.group).toBeGreaterThanOrEqual(
      4.5,
    );
  }
});
