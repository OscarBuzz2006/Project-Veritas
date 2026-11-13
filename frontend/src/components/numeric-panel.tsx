import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { LinePlot } from "@/components/plots";
import { format, humanize, type Benchmark } from "@/lib/results";

const descriptions: Record<string, string> = {
  C1: "Independent baseline",
  C2: "Shared population median and IQR",
  C3: "Shared population median and IQR + persona and conversation history",
};
const palette = ["#000000", "#555555", "#777777"];
const lineStyles = ["solid", "dashed", "dotted"] as const;
const lineDashes = [undefined, "8 5", "2 4"];
const metrics = {
  crowd_error: "Crowd error",
  avg_individual_error: "Average individual error",
  diversity: "Diversity",
  iqr: "Interquartile range",
  miss_rate: "Parse miss rate",
};
type NumericMetric = keyof typeof metrics;

export function NumericPanel({
  numeric,
}: {
  numeric: Benchmark["numeric_analysis"];
}) {
  const [metric, setMetric] = useState<NumericMetric>("crowd_error");
  const finals = Object.values(numeric.by_condition_final);
  return (
    <div className="space-y-8">
      <section className="surface">
        <div className="panel-heading">
          <div>
            <h2>Page's diversity identity</h2>
          </div>
          <Badge variant="outline">Numeric core</Badge>
        </div>
        <p className="numeric-formula">
          Crowd error = Average individual error − Diversity
        </p>
        <p className="panel-description">
          Maximum identity residual:{" "}
          <strong>{numeric.max_identity_residual.toExponential(2)}</strong>.
          This is an algebraic consistency check, not evidence that a real model
          improved.
        </p>
      </section>
      {!finals.length && (
        <p className="empty-state">No numeric results are available.</p>
      )}
      <div className="numeric-cards">
        {finals.map((row) => (
          <section
            className="surface"
            key={row.condition_id}
            aria-label={`Numeric condition ${row.condition_id}`}
          >
            <div className="panel-heading">
              <Badge variant="secondary">{row.condition_id}</Badge>
              <span className="text-xs text-muted-foreground">
                Round {row.round_idx}
              </span>
            </div>
            <p className="panel-description min-h-12">
              {descriptions[row.condition_id] ?? row.condition_id}
            </p>
            <div className="selected-score">{format(row.crowd_error, 6)}</div>
            <p className="text-xs text-muted-foreground">
              Crowd error · lower is better
            </p>
            <dl className="key-values mt-4">
              <div>
                <dt>Individual error</dt>
                <dd>{format(row.avg_individual_error, 6)}</dd>
              </div>
              <div>
                <dt>Diversity</dt>
                <dd>{format(row.diversity, 6)}</dd>
              </div>
            </dl>
          </section>
        ))}
      </div>
      <section className="surface">
        <div className="panel-heading flex-wrap">
          <div>
            <h2>Numeric trajectories</h2>
          </div>
          <div>
            <label htmlFor="numeric-metric" className="field-label">
              Numeric metric
            </label>
            <select
              id="numeric-metric"
              className="control"
              value={metric}
              onChange={(event) =>
                setMetric(event.target.value as NumericMetric)
              }
            >
              {Object.entries(metrics).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="numeric-legend" aria-label="Numeric chart legend">
          {finals.map((row, index) => (
            <span key={row.condition_id}>
              <i
                aria-hidden="true"
                style={{
                  borderTopColor: palette[index % palette.length],
                  borderTopStyle: lineStyles[index % lineStyles.length],
                }}
              />
              {row.condition_id}
            </span>
          ))}
        </div>
        <LinePlot
          label={`${metrics[metric]} by numeric condition and round`}
          series={finals.map((row, index) => ({
            label: row.condition_id,
            color: palette[index % palette.length],
            dash: lineDashes[index % lineDashes.length],
            points: numeric.condition_round_metrics
              .filter((point) => point.condition_id === row.condition_id)
              .sort((a, b) => a.round_idx - b.round_idx)
              .map((point) => ({ x: point.round_idx, y: point[metric] })),
          }))}
        />
        <p className="panel-description">
          Error and diversity use the benchmark's log10 estimate space. Higher
          diversity means wider estimates, not automatically better performance.
          Parse misses are excluded from numeric updates.
        </p>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable numeric scores"
        >
          <table aria-label="Numeric final-round scores">
            <thead>
              <tr>
                <th scope="col">Condition</th>
                <th scope="col">Round</th>
                {Object.values(metrics).map((label) => (
                  <th key={label} scope="col">
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {finals.map((row) => (
                <tr key={row.condition_id}>
                  <th scope="row">{row.condition_id}</th>
                  <td>{row.round_idx}</td>
                  {(Object.keys(metrics) as NumericMetric[]).map((key) => (
                    <td key={key}>{format(row[key], 6)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="surface">
        <h2>Recorded contrasts</h2>
        <p className="panel-description">
          Differences are B − A. Confidence intervals and Holm-adjusted p-values
          are copied from the Python analysis, not recalculated in the browser.
        </p>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable numeric contrasts"
        >
          <table aria-label="Numeric contrasts">
            <thead>
              <tr>
                <th scope="col">Contrast</th>
                <th scope="col">Metric</th>
                <th scope="col">Mean Δ</th>
                <th scope="col">CI low</th>
                <th scope="col">CI high</th>
                <th scope="col">Holm p</th>
                <th scope="col">Cells</th>
              </tr>
            </thead>
            <tbody>
              {numeric.contrasts.map((contrast) => (
                <tr key={contrast.contrast_id}>
                  <th scope="row">
                    {contrast.condition_b} − {contrast.condition_a}
                  </th>
                  <td>{humanize(contrast.metric)}</td>
                  <td>{format(contrast.mean_diff, 6)}</td>
                  <td>{format(contrast.ci_low, 6)}</td>
                  <td>{format(contrast.ci_high, 6)}</td>
                  <td>{format(contrast.holm_adjusted_p, 6)}</td>
                  <td>{contrast.n_cells}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!numeric.contrasts.length && (
          <p className="empty-state">No statistical contrasts were recorded.</p>
        )}
      </section>
    </div>
  );
}
