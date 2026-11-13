import { ComparisonPlot, LinePlot } from "@/components/plots";
import { Button } from "@/components/ui/button";
import {
  atRound,
  colorFor,
  format,
  metrics,
  type Condition,
} from "@/lib/results";

export function Dashboard({
  conditions,
  selected,
  round,
  onSelect,
  onInspect,
  onOverview,
}: {
  conditions: Condition[];
  selected?: Condition;
  round: string;
  onSelect: (id: string) => void;
  onInspect: () => void;
  onOverview: () => void;
}) {
  const score = selected ? atRound(selected, round) : undefined;
  return (
    <div className="dashboard-grid">
      <section className="surface dashboard-map">
        <div className="panel-heading">
          <h2>Diversity × calibration</h2>
          <span className="dashboard-caption">
            {round === "final" ? "Final aggregate" : `Round ${round}`}
          </span>
        </div>
        <ComparisonPlot
          conditions={conditions}
          selected={selected?.condition_id}
          round={round}
          onSelect={onSelect}
        />
      </section>
      <section className="surface dashboard-ledger">
        <div className="panel-heading">
          <h2>Conditions</h2>
          <Button variant="ghost" onClick={onOverview}>
            All metrics
          </Button>
        </div>
        <p className="dashboard-caption">
          Select a row to inspect its scores and trajectory.
        </p>
        <div
          className="dashboard-table"
          role="region"
          aria-label="Dashboard condition scores"
          tabIndex={0}
        >
          <table aria-label="Dashboard scores">
            <thead>
              <tr>
                <th scope="col">Condition</th>
                <th scope="col">Topics ↑</th>
                <th scope="col">Brier ↓</th>
                <th scope="col">Accuracy ↑</th>
              </tr>
            </thead>
            <tbody>
              {conditions.map((condition, index) => {
                const values = atRound(condition, round);
                return (
                  <tr
                    key={condition.condition_id}
                    data-selected={
                      selected?.condition_id === condition.condition_id
                    }
                  >
                    <th scope="row">
                      <button
                        type="button"
                        className="dashboard-condition"
                        aria-pressed={
                          selected?.condition_id === condition.condition_id
                        }
                        onClick={() => onSelect(condition.condition_id)}
                        title={condition.label}
                      >
                        <span
                          className="condition-number"
                          style={{ background: colorFor(condition) }}
                        >
                          {index + 1}
                        </span>
                        <span>{condition.label}</span>
                      </button>
                    </th>
                    <td>{format(values?.effective_topic_count, 2)}</td>
                    <td>{format(values?.brier_score)}</td>
                    <td>{format(values?.ground_truth_accuracy, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!conditions.length && (
            <p className="empty-state">
              No matching conditions. Reset the sidebar filters to see the run.
            </p>
          )}
        </div>
      </section>
      <section
        className="surface dashboard-selection"
        aria-label="Selected condition summary"
      >
        {selected ? (
          <>
            <div className="panel-heading">
              <h2 title={selected.label}>{selected.label}</h2>
              <Button variant="outline" onClick={onInspect}>
                Inspect responses
              </Button>
            </div>
            <div className="dashboard-selection-body">
              <dl className="dashboard-metrics">
                {Object.entries(metrics).map(([key, info]) => (
                  <div key={key}>
                    <dt>{info.label}</dt>
                    <dd>{format(score?.[key as keyof typeof metrics])}</dd>
                  </div>
                ))}
              </dl>
              <div className="dashboard-trend">
                <p className="dashboard-caption">
                  Effective topics across rounds
                </p>
                <LinePlot
                  label={`${selected.label}: topic trajectory`}
                  series={[
                    {
                      label: selected.label,
                      color: colorFor(selected),
                      points: selected.round_trajectory.map((point) => ({
                        x: point.round_idx,
                        y: point.effective_topic_count,
                      })),
                    },
                  ]}
                />
              </div>
            </div>
          </>
        ) : (
          <p className="empty-state">Select a condition to see its metrics.</p>
        )}
      </section>
    </div>
  );
}
