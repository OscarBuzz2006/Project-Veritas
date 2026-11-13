import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import type { Benchmark } from "../src/lib/results";

const benchmark: Benchmark = JSON.parse(
  readFileSync(
    new URL("../../docs/data/benchmark_summary.json", import.meta.url),
    "utf8",
  ),
);

test("personalized long model badge and expanded raw record stay within the viewport", async ({
  page,
}) => {
  const specimen = benchmark.trace_specimens.find(
    (record) =>
      record.model_assigned === "anthropic/claude-frontier@persona-agent_01",
  )!;
  expect(specimen).toBeDefined();
  const index = benchmark.trace_specimens
    .filter((record) => record.condition_id === specimen.condition_id)
    .indexOf(specimen);
  await page.goto("/");
  await page
    .getByRole("button", { name: "Inspect responses", exact: true })
    .click();
  await page
    .getByLabel("Trace condition", { exact: true })
    .selectOption(specimen.condition_id);
  const article = page.getByRole("article", {
    name: `Specimen ${index + 1}: ${specimen.agent_id}`,
    exact: true,
  });
  const badge = article.getByText(specimen.model_assigned, { exact: true });
  await expect(badge).toBeVisible();
  await expect(badge).toHaveCSS("white-space", "normal");
  await expect(badge).toHaveCSS("overflow-wrap", "anywhere");
  await article
    .getByRole("button", {
      name: `Raw record · specimen ${index + 1}`,
      exact: true,
    })
    .click();
  const raw = article.getByLabel(`Raw JSON specimen ${index + 1}`, {
    exact: true,
  });
  await expect(raw).toBeVisible();
  await expect(raw).toContainText(specimen.model_assigned);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  for (const element of [badge, raw]) {
    await expect
      .poll(() =>
        element.evaluate((node) => {
          const bounds = node.getBoundingClientRect();
          return (
            bounds.left >= 0 &&
            bounds.right <= window.innerWidth &&
            node.scrollWidth <= node.clientWidth
          );
        }),
      )
      .toBe(true);
  }
});

test("explore recorded scores, numeric results, and shadcn response traces", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Results explorer", exact: true }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("table", { name: "Condition scores", exact: true })
      .getByRole("row"),
  ).toHaveCount(benchmark.summaries.length + 1);
  await expect(page.locator("body")).toContainText(
    "Synthetic runs demonstrate simulator assumptions",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("overview.png"),
    fullPage: true,
  });

  await page
    .getByLabel("Condition family", { exact: true })
    .selectOption("baseline");
  await page
    .getByLabel("Communication round", { exact: true })
    .selectOption("0");
  await expect(
    page.getByRole("table", { name: "Condition scores", exact: true }),
  ).toContainText(
    benchmark.summaries[0].round_trajectory[0].brier_score.toFixed(3),
  );
  await page
    .getByRole("button", { name: "Inspect responses", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Recorded response specimens" }),
  ).toBeVisible();
  await expect(page.locator("[data-slot=message]")).toHaveCount(
    benchmark.trace_specimens.filter(
      (record) => record.condition_id === "no_ai",
    ).length * 2,
  );
  const context = page.getByRole("button", {
    name: "Recorded context · specimen 1",
    exact: true,
  });
  await context.focus();
  await page.keyboard.press("Enter");
  await expect(context).toHaveAttribute("aria-expanded", "true");
  await expect(
    page.getByText(benchmark.trace_specimens[0].model_family, { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("responses.png"),
    fullPage: true,
  });

  await page.getByRole("tab", { name: "Numeric core", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Page's diversity identity",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByLabel("Numeric metric", { exact: true })
    .selectOption("diversity");
  await expect(
    page.getByRole("img", {
      name: "Diversity by numeric condition and round",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("numeric.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("round navigation, removable filters, and visible CSV stay in sync", async ({
  page,
}) => {
  await page.goto("/");
  const family = page.getByLabel("Condition family", { exact: true });
  await family.selectOption("baseline");
  await page
    .getByRole("button", { name: "Previous round", exact: true })
    .click();
  const lastRound = Math.max(
    ...benchmark.summaries.flatMap((row) =>
      row.round_trajectory.map((point) => point.round_idx),
    ),
  );
  await expect(
    page.getByLabel("Communication round", { exact: true }),
  ).toHaveValue(String(lastRound));
  await page
    .getByLabel("Communication round", { exact: true })
    .selectOption("0");
  await expect(
    page.getByRole("button", { name: "Previous round", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Next round", exact: true }).click();
  await expect(
    page.getByLabel("Communication round", { exact: true }),
  ).toHaveValue("1");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export visible CSV" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe("epistemic-prism-scores-1.csv");
  const csv = readFileSync((await download.path())!, "utf8");
  const baseline = benchmark.summaries.filter(
    (row) => row.group === "baseline",
  );
  expect(csv.split("\r\n")).toHaveLength(baseline.length + 1);
  expect(csv).toContain(
    String(
      baseline[0].round_trajectory.find((point) => point.round_idx === 1)!
        .brier_score,
    ),
  );
  await page.getByRole("button", { name: "Remove family filter" }).click();
  await expect(family).toHaveValue("all");
  await page
    .getByRole("button", { name: "Reset round to final aggregate" })
    .click();
  await expect(
    page.getByRole("button", { name: "Next round", exact: true }),
  ).toBeDisabled();
  await expect(page.getByLabel("Active filters", { exact: true })).toHaveCount(
    0,
  );
  await expect(family).toHaveCSS("color-scheme", "dark");
  await expect(family.locator("option").first()).toHaveCSS(
    "background-color",
    "rgb(48, 48, 48)",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("trace layout separates labels and avatars and exposes baseline setup", async ({
  page,
}, info) => {
  await page.goto("/");
  await page
    .getByLabel("Condition family", { exact: true })
    .selectOption("personalization");
  await page.getByRole("tab", { name: "Response traces", exact: true }).click();
  await page
    .getByRole("button", { name: "View baseline traces", exact: true })
    .click();
  await expect(page.getByLabel("Trace condition", { exact: true })).toHaveValue(
    "no_ai",
  );
  await expect(
    page.getByRole("button", { name: "Viewing baseline" }),
  ).toBeDisabled();
  const article = page.getByRole("article").first();
  const label = await article.locator(".specimen-label").boundingBox();
  const avatar = await article
    .locator('[data-slot="message-avatar"]')
    .first()
    .boundingBox();
  expect(avatar!.y).toBeGreaterThanOrEqual(label!.y + label!.height);
  await article
    .getByRole("button", { name: "System prompts · specimen 1", exact: true })
    .click();
  await expect(
    article.getByText(
      "Not applicable: this baseline has no AI assistant in the loop.",
    ),
  ).toBeVisible();
  await article
    .getByRole("button", { name: "Recorded context · specimen 1", exact: true })
    .click();
  await expect(
    article.getByText("Agent region", { exact: true }),
  ).toBeVisible();
  await expect(article.getByText("Task", { exact: true })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await article.screenshot({
    path: info.outputPath("trace-detail.png"),
    animations: "disabled",
  });
});

test("compact dashboard fits a laptop screen and keeps selection linked", async ({
  page,
}, info) => {
  if (info.project.name === "desktop")
    await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto("/");
  await page.getByRole("tab", { name: "Dashboard", exact: true }).click();
  await expect(
    page
      .getByRole("table", { name: "Dashboard scores", exact: true })
      .getByRole("row"),
  ).toHaveCount(benchmark.summaries.length + 1);
  const chosen = benchmark.summaries[2];
  await page
    .getByRole("button", { name: `3 ${chosen.label}`, exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "Selected condition summary" }),
  ).toContainText(chosen.label);
  if (info.project.name === "desktop") {
    expect(
      await page.evaluate(
        () => document.documentElement.scrollHeight <= innerHeight,
      ),
    ).toBe(true);
    const ledger = page.locator(".dashboard-table");
    expect(
      await ledger.evaluate((el) => el.scrollHeight <= el.clientHeight),
    ).toBe(true);
    const summary = await page
      .getByRole("region", { name: "Selected condition summary" })
      .boundingBox();
    expect(summary!.y + summary!.height).toBeLessThanOrEqual(768);
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: info.outputPath("dashboard.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Inspect responses", exact: true })
    .click();
  await expect(page.getByLabel("Trace condition", { exact: true })).toHaveValue(
    chosen.condition_id,
  );
});
