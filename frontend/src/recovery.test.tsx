import { readFileSync } from "node:fs";
import { screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";

vi.mock("./App", () => ({
  default: () => {
    throw new Error("Invalid report fixture");
  },
}));

it("offers the deployed legacy report when rendering fails", async () => {
  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
  const root = document.createElement("div");
  root.id = "root";
  document.body.append(root);
  try {
    await import("./main");
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    const link = screen.getByRole("link", {
      name: /standalone Python report/i,
    });
    expect(link.getAttribute("href")).toBe("./legacy/");
  } finally {
    root.remove();
    consoleError.mockRestore();
  }
});

it("offers the same deployed fallback without JavaScript", () => {
  const html = new DOMParser().parseFromString(
    readFileSync("index.html", "utf8"),
    "text/html",
  );
  const link = html.querySelector("noscript a");
  expect(link).not.toBeNull();
  expect(link!.getAttribute("href")).toBe("./legacy/");
});
