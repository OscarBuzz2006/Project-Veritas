import { atRound, colorFor, format, type Condition } from "@/lib/results";

export function ComparisonPlot({
  conditions,
  round,
  selected,
  onSelect,
}: {
  conditions: Condition[];
  round: string;
  selected?: string;
  onSelect: (id: string) => void;
}) {
  const points = conditions.flatMap((condition, index) => {
    const score = atRound(condition, round);
    return score ? [{ condition, score, index }] : [];
  });
  const maxX =
    Math.max(1, ...points.map(({ score }) => score.effective_topic_count)) *
    1.15;
  const maxY =
    Math.max(0.5, ...points.map(({ score }) => score.brier_score)) * 1.1;
  const x = (value: number) => 58 + (value / maxX) * 530;
  const y = (value: number) => 35 + (value / maxY) * 220;
  return (
    <div
      className="chart-scroll"
      tabIndex={0}
      role="region"
      aria-label="Diversity and calibration chart"
    >
      <svg
        viewBox="0 0 630 310"
        role="group"
        aria-label="Condition comparison; select a point to inspect it"
      >
        <text x="58" y="17" className="axis-caption">
          Brier score · lower is better ↑
        </text>
        {[0, 1, 2, 3, 4].map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1="58"
              x2="588"
              y1={y((maxY * tick) / 4)}
              y2={y((maxY * tick) / 4)}
            />
            <text
              className="tick"
              x="48"
              y={y((maxY * tick) / 4) + 4}
              textAnchor="end"
            >
              {format((maxY * tick) / 4, 2)}
            </text>
            <text
              className="tick"
              x={x((maxX * tick) / 4)}
              y="278"
              textAnchor="middle"
            >
              {format((maxX * tick) / 4, 1)}
            </text>
          </g>
        ))}
        <text x="323" y="304" className="axis-caption" textAnchor="middle">
          Effective topics · broader coverage →
        </text>
        {points.map(({ condition, score, index }) => (
          <g key={condition.condition_id}>
            <circle
              cx={x(score.effective_topic_count)}
              cy={y(score.brier_score)}
              r={selected === condition.condition_id ? 12 : 9}
              fill={colorFor(condition)}
              stroke={selected === condition.condition_id ? "#171717" : "white"}
              strokeWidth="2"
              role="button"
              tabIndex={0}
              aria-label={`Select ${condition.label}`}
              aria-pressed={selected === condition.condition_id}
              onClick={() => onSelect(condition.condition_id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(condition.condition_id);
                }
              }}
            >
              <title>
                {condition.label}: {format(score.effective_topic_count)} topics,
                Brier {format(score.brier_score)}
              </title>
            </circle>
            <text
              x={x(score.effective_topic_count)}
              y={y(score.brier_score) + 3}
              className="point-label"
              pointerEvents="none"
              textAnchor="middle"
            >
              {index + 1}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export type Series = {
  label: string;
  color: string;
  dash?: string;
  points: { x: number; y: number }[];
};
export function LinePlot({
  series,
  label,
}: {
  series: Series[];
  label: string;
}) {
  const points = series
    .flatMap((line) => line.points)
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points.length)
    return (
      <p className="empty-state">No recorded rounds for this selection.</p>
    );
  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(minX + 1, ...points.map((p) => p.x));
  const minY = Math.min(0, ...points.map((p) => p.y));
  const maxY = Math.max(0.001, ...points.map((p) => p.y)) * 1.12;
  const x = (value: number) => 62 + ((value - minX) / (maxX - minX)) * 510;
  const y = (value: number) => 245 - ((value - minY) / (maxY - minY)) * 215;
  return (
    <div className="chart-scroll" tabIndex={0} role="region" aria-label={label}>
      <svg viewBox="0 0 620 300" role="img" aria-label={label}>
        <title>{label}</title>
        {[0, 1, 2, 3, 4].map((tick) => {
          const value = minY + ((maxY - minY) * tick) / 4;
          return (
            <g key={tick}>
              <line
                className="grid-line"
                x1="62"
                x2="572"
                y1={y(value)}
                y2={y(value)}
              />
              <text className="tick" x="52" y={y(value) + 4} textAnchor="end">
                {format(value, maxY < 0.1 ? 3 : 2)}
              </text>
            </g>
          );
        })}
        {[...new Set(points.map((p) => p.x))]
          .sort((a, b) => a - b)
          .map((round) => (
            <text
              key={round}
              className="tick"
              x={x(round)}
              y="266"
              textAnchor="middle"
            >
              {round}
            </text>
          ))}
        <text x="317" y="292" className="axis-caption" textAnchor="middle">
          Communication round
        </text>
        {series.map((line) => (
          <g key={line.label}>
            <polyline
              fill="none"
              stroke={line.color}
              strokeWidth="2.5"
              strokeDasharray={line.dash}
              points={line.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
            />
            {line.points.map((point) => (
              <circle
                key={point.x}
                cx={x(point.x)}
                cy={y(point.y)}
                r="4"
                fill={line.color}
                stroke="white"
                strokeWidth="1.5"
              >
                <title>
                  {line.label}, round {point.x}: {format(point.y, 6)}
                </title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
    </div>
  );
}
