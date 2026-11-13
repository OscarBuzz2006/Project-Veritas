import type benchmark from "../../../docs/data/benchmark_summary.json";

type NumericCondition =
  (typeof benchmark)["numeric_analysis"]["by_condition_final"]["C1"];
export type Benchmark = Omit<typeof benchmark, "numeric_analysis"> & {
  numeric_analysis: Omit<
    (typeof benchmark)["numeric_analysis"],
    "by_condition_final"
  > & {
    by_condition_final: Record<string, NumericCondition>;
  };
};
export type Condition = Benchmark["summaries"][number];
export type Specimen = Benchmark["trace_specimens"][number];
export type Metric =
  | "effective_topic_count"
  | "brier_score"
  | "ground_truth_accuracy"
  | "goldman_effective_sources"
  | "confident_wrong_rate"
  | "pairwise_jaccard_distance";

export const metrics: Record<Metric, { label: string; direction: string }> = {
  effective_topic_count: {
    label: "Effective topics",
    direction: "Higher means broader topic coverage",
  },
  brier_score: { label: "Brier score", direction: "Lower Brier is better" },
  ground_truth_accuracy: { label: "Accuracy", direction: "Higher is better" },
  goldman_effective_sources: {
    label: "Effective sources",
    direction: "Higher means less source redundancy",
  },
  confident_wrong_rate: {
    label: "Confident-wrong rate",
    direction: "Lower is better",
  },
  pairwise_jaccard_distance: {
    label: "Lexical distance",
    direction:
      "Higher means different wording, not necessarily different ideas",
  },
};
export const groupLabels: Record<string, string> = {
  baseline: "Baseline",
  decoding_ablation: "Decoding",
  k_sweep: "Model count",
  nationality: "Nationality",
  personalization: "Personalization",
  main: "Main",
};
const colors: Record<string, string> = {
  baseline: "#555555",
  decoding_ablation: "#333333",
  k_sweep: "#666666",
  nationality: "#444444",
  personalization: "#222222",
  main: "#111111",
};
export const colorFor = (condition: Condition) =>
  colors[condition.group] ?? "#525252";
export const humanize = (value: string) => value.replaceAll("_", " ");
export const format = (value: number | undefined, digits = 3) =>
  value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
export function atRound(condition: Condition, round: string) {
  return round === "final"
    ? condition
    : condition.round_trajectory.find(
        (point) => point.round_idx === Number(round),
      );
}
