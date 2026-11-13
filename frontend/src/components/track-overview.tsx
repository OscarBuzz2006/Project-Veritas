import { useState } from "react";
import { ArrowDownUp, ScanSearch, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ComparisonPlot, LinePlot } from "@/components/plots";
import {
  atRound,
  colorFor,
  format,
  metrics,
  type Condition,
  type Metric,
} from "@/lib/results";

export function TrackOverview({
  conditions,
  round,
  selected,
  onSelect,
  onInspect,
}: {
  conditions: Condition[];
  round: string;
  selected?: Condition;
  onSelect: (id: string) => void;
  onInspect: () => void;
}) {
  const [metric, setMetric] = useState<Metric>("effective_topic_count");
  const [sort, setSort] = useState<{ key: Metric; asc: boolean }>({
    key: "effective_topic_count",
    asc: false,
  });
  const rows = [...conditions].sort((a, b) => {
    const av = atRound(a, round)?.[sort.key],
      bv = atRound(b, round)?.[sort.key];
    if (av === undefined) return bv === undefined ? 0 : 1;
    if (bv === undefined) return -1;
    return (av - bv) * (sort.asc ? 1 : -1);
  });
  const columns: Metric[] = [
    "effective_topic_count",
    "brier_score",
    "ground_truth_accuracy",
    "goldman_effective_sources",
    "confident_wrong_rate",
    "pairwise_jaccard_distance",
  ];
  function exportScores() {
    const csvRows = [
      [
        "Condition",
        "Condition ID",
        "Round",
        ...columns.map((key) => metrics[key].label),
      ],
      ...rows.map((condition) => [
        condition.label,
        condition.condition_id,
        round,
        ...columns.map((key) => atRound(condition, round)?.[key] ?? ""),
      ]),
    ];
    const csv = csvRows
      .map((row) =>
        row
          .map((value) => `"${String(value).replaceAll('"', '""')}"`)
          .join(","),
      )
      .join("\r\n");
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `epistemic-prism-scores-${round}.csv`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <div className="space-y-8">
      <div className="comparison-grid">
        <section className="surface">
          <div className="panel-heading">
            <div>
              <h2>Diversity × calibration</h2>
            </div>
          </div>
          <p className="panel-description">
            Effective topics and Brier score across recorded conditions.
          </p>
          <ComparisonPlot
            conditions={conditions}
            round={round}
            selected={selected?.condition_id}
            onSelect={onSelect}
          />
          <div className="chart-footnote">
            Equal-sized points represent conditions. Upper right means broader
            topics and lower error. Exact values are in the score table.
          </div>
        </section>
        <section
          className="surface condition-list"
          aria-label="Conditions in chart"
        >
          <div className="panel-heading">
            <h2>Conditions</h2>
            <Badge variant="secondary">{conditions.length}</Badge>
          </div>
          <div className="condition-options">
            {conditions.map((condition, index) => (
              <button
                key={condition.condition_id}
                type="button"
                className="condition-option"
                aria-pressed={selected?.condition_id === condition.condition_id}
                onClick={() => onSelect(condition.condition_id)}
              >
                <span
                  className="condition-number"
                  style={{ background: colorFor(condition) }}
                >
                  {index + 1}
                </span>
                <span className="min-w-0">
                  <strong>{condition.label}</strong>
                  <span>{condition.condition_id}</span>
                </span>
              </button>
            ))}
          </div>
        </section>
      </div>
      {selected && (
        <section className="surface">
          <div className="panel-heading flex-wrap gap-3">
            <div>
              <span className="eyebrow">Round trajectory</span>
              <h2>{selected.label}</h2>
            </div>
            <Button variant="outline" onClick={onInspect}>
              <ScanSearch aria-hidden="true" />
              Inspect responses
            </Button>
          </div>
          <div className="trajectory-grid">
            <div>
              <label className="field-label" htmlFor="trajectory-metric">
                Trajectory metric
              </label>
              <select
                id="trajectory-metric"
                className="control max-w-64"
                value={metric}
                onChange={(event) => setMetric(event.target.value as Metric)}
              >
                {Object.entries(metrics).map(([key, info]) => (
                  <option key={key} value={key}>
                    {info.label}
                  </option>
                ))}
              </select>
              <p className="panel-description mt-3">
                {metrics[metric].direction}.
              </p>
              <div className="selected-score">
                {format(atRound(selected, round)?.[metric])}
              </div>
              <p className="text-xs text-muted-foreground">
                {round === "final" ? "Final aggregate" : `Round ${round}`} ·{" "}
                {metrics[metric].label.toLowerCase()}
              </p>
              <dl className="key-values mt-5">
                <div>
                  <dt>Models</dt>
                  <dd>{selected.num_models}</dd>
                </div>
                <div>
                  <dt>Temperature</dt>
                  <dd>{selected.temperature}</dd>
                </div>
                <div>
                  <dt>Top-p</dt>
                  <dd>{selected.top_p}</dd>
                </div>
              </dl>
            </div>
            <LinePlot
              label={`${selected.label}: ${metrics[metric].label} by round`}
              series={[
                {
                  label: selected.label,
                  color: colorFor(selected),
                  points: selected.round_trajectory.map((point) => ({
                    x: point.round_idx,
                    y: point[metric],
                  })),
                },
              ]}
            />
          </div>
        </section>
      )}
      <section className="surface">
        <div className="panel-heading">
          <div>
            <h2>Condition scores</h2>
          </div>
          <div className="score-actions">
            <Badge variant="outline">
              {round === "final" ? "Final aggregate" : `Round ${round}`}
            </Badge>
            <Button
              variant="outline"
              onClick={exportScores}
              disabled={!rows.length}
            >
              <Download aria-hidden="true" />
              Export visible CSV
            </Button>
          </div>
        </div>
        <p className="panel-description">
          Lower Brier is better. Sort a column, then inspect its recorded
          responses.
        </p>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable condition scores"
        >
          <table aria-label="Condition scores">
            <thead>
              <tr>
                <th scope="col">Condition</th>
                {columns.map((key) => (
                  <th
                    key={key}
                    scope="col"
                    aria-sort={
                      sort.key === key
                        ? sort.asc
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <button
                      type="button"
                      onClick={() =>
                        setSort({
                          key,
                          asc:
                            sort.key === key
                              ? !sort.asc
                              : key === "brier_score" ||
                                key === "confident_wrong_rate",
                        })
                      }
                    >
                      {metrics[key].label}
                      <ArrowDownUp className="size-3" aria-hidden="true" />
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((condition) => (
                <tr
                  key={condition.condition_id}
                  data-selected={
                    selected?.condition_id === condition.condition_id
                  }
                >
                  <th scope="row">
                    <button
                      className="table-condition"
                      type="button"
                      onClick={() => {
                        onSelect(condition.condition_id);
                        onInspect();
                      }}
                    >
                      {condition.label}
                      <span>{condition.condition_id}</span>
                    </button>
                  </th>
                  {columns.map((key) => (
                    <td key={key}>
                      {format(atRound(condition, round)?.[key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
