import { useState } from "react";
import {
  CircleHelp,
  Database,
  LayoutDashboard,
  MessageSquare,
  Search,
  Sigma,
  SlidersHorizontal,
  RotateCcw,
  Download,
  ChevronLeft,
  ChevronRight,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TrackOverview } from "@/components/track-overview";
import { ResponseInspector } from "@/components/response-inspector";
import { NumericPanel } from "@/components/numeric-panel";
import { Dashboard } from "@/components/dashboard";
import { format, groupLabels, type Benchmark } from "@/lib/results";

export default function App({ data }: { data: Benchmark }) {
  const [tab, setTab] = useState("overview");
  const [group, setGroup] = useState("all");
  const [query, setQuery] = useState("");
  const [round, setRound] = useState("final");
  const [selectedId, setSelectedId] = useState(data.summaries[0]?.condition_id);
  const groups = [...new Set(data.summaries.map((item) => item.group))];
  const rounds = [
    ...new Set(
      data.summaries.flatMap((item) =>
        item.round_trajectory.map((point) => point.round_idx),
      ),
    ),
  ].sort((a, b) => a - b);
  const roundChoices = [...rounds.map(String), "final"];
  const roundPosition = roundChoices.indexOf(round);
  const hasFilters =
    group !== "all" || query.trim() !== "" || round !== "final";
  const conditions = data.summaries.filter(
    (item) =>
      (group === "all" || item.group === group) &&
      `${item.label} ${item.condition_id}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
  );
  const selected =
    conditions.find((item) => item.condition_id === selectedId) ??
    conditions[0];
  function resetFilters() {
    setGroup("all");
    setQuery("");
    setRound("final");
  }
  function downloadResults() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "epistemic-prism-results.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <Tabs
      value={tab}
      onValueChange={setTab}
      orientation="vertical"
      className="block"
    >
      <div
        className={`app-shell${tab === "dashboard" ? " dashboard-mode" : ""}`}
      >
        <a className="skip-link" href="#results-main">
          Skip to results
        </a>
        <aside className="workspace-sidebar" aria-label="Workspace and filters">
          <a className="brand" href="#results-main">
            <svg
              className="prism-symbol"
              viewBox="0 0 40 40"
              width="38"
              height="38"
              fill="none"
              aria-hidden="true"
            >
              <path d="M20 4 37 33H3Z" stroke="#ffffff" strokeWidth="1.8" />
              <path
                d="m20 4 4 29M3 33l17-14 17 14M20 19V4"
                stroke="#ffffff"
                strokeWidth="1.2"
              />
              <path d="m20 19 4 14h13Z" fill="#ffffff" fillOpacity=".65" />
            </svg>
            <span>
              Epistemic Prism<small>Multi-agent benchmark</small>
            </span>
          </a>
          <TabsList
            variant="line"
            aria-label="Results views"
            className="sidebar-navigation"
          >
            <TabsTrigger value="dashboard">
              <LayoutDashboard aria-hidden="true" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="overview">
              <LayoutDashboard aria-hidden="true" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="numeric">
              <Sigma aria-hidden="true" />
              Numeric core
            </TabsTrigger>
            <TabsTrigger value="traces">
              <MessageSquare aria-hidden="true" />
              Response traces
            </TabsTrigger>
          </TabsList>
          <div className="sidebar-section">
            <h2 className="eyebrow">
              <SlidersHorizontal size={14} aria-hidden="true" /> Track 4 filters
            </h2>
            <label className="field-label" htmlFor="condition-query">
              Search conditions
            </label>
            <div className="search-control">
              <Search className="size-4" aria-hidden="true" />
              <input
                id="condition-query"
                type="search"
                value={query}
                placeholder="Find a condition…"
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <label className="field-label mt-5" htmlFor="condition-group">
              Condition family
            </label>
            <select
              id="condition-group"
              className="control"
              value={group}
              onChange={(event) => setGroup(event.target.value)}
            >
              <option value="all">All families</option>
              {groups.map((key) => (
                <option key={key} value={key}>
                  {groupLabels[key] ?? key}
                </option>
              ))}
            </select>
            <label className="field-label mt-5" htmlFor="communication-round">
              Communication round
            </label>
            <select
              id="communication-round"
              className="control"
              value={round}
              onChange={(event) => setRound(event.target.value)}
            >
              <option value="final">Final aggregate</option>
              {rounds.map((value) => (
                <option key={value} value={value}>
                  Round {value}
                </option>
              ))}
            </select>
            <div
              className="round-navigation"
              role="group"
              aria-label="Navigate communication rounds"
            >
              <button
                type="button"
                aria-label="Previous round"
                disabled={roundPosition <= 0}
                onClick={() => setRound(roundChoices[roundPosition - 1])}
              >
                <ChevronLeft size={16} aria-hidden="true" />
              </button>
              <span>
                {round === "final" ? "Final aggregate" : `Round ${round}`}
              </span>
              <button
                type="button"
                aria-label="Next round"
                disabled={roundPosition >= roundChoices.length - 1}
                onClick={() => setRound(roundChoices[roundPosition + 1])}
              >
                <ChevronRight size={16} aria-hidden="true" />
              </button>
            </div>
            <Button
              variant="ghost"
              className="mt-3 w-full justify-start"
              onClick={resetFilters}
            >
              <RotateCcw size={14} aria-hidden="true" />
              Reset filters
            </Button>
          </div>
          <div className="sidebar-bottom">
            <strong>About these filters</strong>
            <p>
              Search and family filters apply to Track 4 conditions. Numeric
              results use their own controls.
            </p>
          </div>
        </aside>
        <main
          id="results-main"
          className="workspace-main"
          aria-label={data.metadata.title}
        >
          <header className="workspace-topbar">
            <span>
              Apart Research <span className="mx-2 text-border">/</span>{" "}
              <strong>Benchmark</strong>
            </span>
            <Badge variant="outline">
              <Database className="size-3" aria-hidden="true" />
              Saved results
            </Badge>
          </header>
          <div className="page-content">
            <div className="page-title">
              <div>
                <h1>Results explorer</h1>
                <p>
                  Compare conditions, inspect responses, and check calibration.
                </p>
              </div>
              <Button
                variant="outline"
                onClick={downloadResults}
                className="export-button"
              >
                <Download aria-hidden="true" />
                Export results
              </Button>
            </div>
            <div className="evidence-note">
              <CircleHelp className="size-4 shrink-0" aria-hidden="true" />
              <p>
                <strong>Interpretation note.</strong> Synthetic runs demonstrate
                simulator assumptions, not real-model findings. Verify
                provenance before interpreting a run.
              </p>
            </div>
            <div className="stats-grid" aria-label="Run summary">
              {[
                ["Conditions", data.summaries.length],
                ["Agent-round records", data.metadata.n_track4_records],
                ["Numeric records", data.metadata.n_numeric_records],
                ["Trace specimens", data.trace_specimens.length],
              ].map(([label, value]) => (
                <div className="stat" key={label}>
                  <span>{label}</span>
                  <strong>
                    {typeof value === "number"
                      ? value.toLocaleString("en-US")
                      : value}
                  </strong>
                </div>
              ))}
            </div>
            <div className="workspace-tabs">
              <div className="tabs-toolbar">
                <span className="view-status" role="status">
                  {tab === "numeric"
                    ? "Numeric conditions · independent of Track 4 filters"
                    : `${conditions.length} of ${data.summaries.length} conditions`}
                </span>
              </div>
              {tab !== "numeric" && tab !== "dashboard" && hasFilters && (
                <div className="active-filters" aria-label="Active filters">
                  <span>
                    Showing {conditions.length} of {data.summaries.length}{" "}
                    conditions
                  </span>
                  {group !== "all" && (
                    <button
                      className="filter-chip"
                      type="button"
                      aria-label="Remove family filter"
                      onClick={() => setGroup("all")}
                    >
                      <span>{groupLabels[group] ?? group}</span>
                      <X size={13} aria-hidden="true" />
                    </button>
                  )}
                  {query.trim() && (
                    <button
                      className="filter-chip"
                      type="button"
                      aria-label="Remove search filter"
                      onClick={() => setQuery("")}
                    >
                      <span>Search: {query.trim()}</span>
                      <X size={13} aria-hidden="true" />
                    </button>
                  )}
                  {tab === "overview" && round !== "final" && (
                    <button
                      className="filter-chip"
                      type="button"
                      aria-label="Reset round to final aggregate"
                      onClick={() => setRound("final")}
                    >
                      <span>Round {round}</span>
                      <X size={13} aria-hidden="true" />
                    </button>
                  )}
                </div>
              )}
              <TabsContent value="dashboard" className="dashboard-tab">
                <Dashboard
                  conditions={conditions}
                  selected={selected}
                  round={round}
                  onSelect={setSelectedId}
                  onInspect={() => setTab("traces")}
                  onOverview={() => setTab("overview")}
                />
              </TabsContent>
              <TabsContent value="overview">
                {!conditions.length && (
                  <div className="empty-state">
                    <Search size={24} aria-hidden="true" />
                    <h2>No conditions match these filters.</h2>
                    <p>
                      Try another search or reset the filters to see every
                      condition.
                    </p>
                    <Button variant="outline" onClick={resetFilters}>
                      Reset search and filters
                    </Button>
                  </div>
                )}
                <TrackOverview
                  conditions={conditions}
                  round={round}
                  selected={selected}
                  onSelect={setSelectedId}
                  onInspect={() => setTab("traces")}
                />
              </TabsContent>
              <TabsContent value="numeric">
                <NumericPanel numeric={data.numeric_analysis} />
              </TabsContent>
              <TabsContent value="traces">
                <ResponseInspector
                  conditions={conditions}
                  selected={selected}
                  specimens={data.trace_specimens}
                  onSelect={setSelectedId}
                  baselineConditions={data.summaries.filter(
                    (condition) => condition.group === "baseline",
                  )}
                  onSelectBaseline={(id) => {
                    setGroup("all");
                    setQuery("");
                    setSelectedId(id);
                  }}
                />
              </TabsContent>
            </div>
            <details className="run-details">
              <summary>Run metadata & interpretation notes</summary>
              <div className="details-grid">
                <div>
                  <h2>Topic-label audit</h2>
                  <p>
                    Judge agreement κ ={" "}
                    {format(data.topic_audit.inter_rater_kappa)} across{" "}
                    {data.topic_audit.n_evaluated.toLocaleString("en-US")}{" "}
                    evaluated records.
                  </p>
                  <p>
                    {data.topic_audit.reliability_warning
                      ? "Warning: topic-label reliability is below threshold."
                      : "Synthetic agreement does not replace independent human validation."}
                  </p>
                </div>
                <div>
                  <h2>Data source</h2>
                  <p>docs/data/benchmark_summary.json</p>
                  <p>
                    Scores are imported from the Python report. No live
                    requests, invented reasoning, or hidden tool activity.
                  </p>
                </div>
              </div>
            </details>
            <footer className="workspace-footer">
              <span>Epistemic Prism · Apart Research</span>
              {import.meta.env.PROD ? (
                <a href="./legacy/" className="underline underline-offset-4">
                  Open standalone report
                </a>
              ) : (
                <span>Standalone report: make serve</span>
              )}
            </footer>
          </div>
        </main>
      </div>
    </Tabs>
  );
}
