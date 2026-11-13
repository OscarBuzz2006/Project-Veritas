import { Component, StrictMode, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import benchmark from "../../docs/data/benchmark_summary.json";
import App from "./App";
import "./index.css";

class ReportBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(
      "Results explorer failed to render",
      error,
      info.componentStack,
    );
  }
  render() {
    return this.state.failed ? (
      <main className="surface m-6" role="alert">
        <h1>Could not display this report</h1>
        <p>
          Check the benchmark_summary.json schema and the browser console. The{" "}
          <a href="./legacy/" className="underline">
            standalone Python report
          </a>{" "}
          remains available.
        </p>
      </main>
    ) : (
      this.props.children
    );
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ReportBoundary>
      <App data={benchmark} />
    </ReportBoundary>
  </StrictMode>,
);
