import {
  Bot,
  Braces,
  FileText,
  Fingerprint,
  UserRound,
  Wrench,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import {
  Message,
  MessageAvatar,
  MessageContent,
  MessageFooter,
  MessageGroup,
  MessageHeader,
} from "@/components/ui/message";
import {
  Trace,
  TraceCollapsible,
  TraceCollapsibleContent,
  TraceCollapsibleTrigger,
  TraceIcon,
  TraceLine,
  TraceTitle,
} from "@/components/ui/trace";
import { format, humanize, type Condition, type Specimen } from "@/lib/results";

function SpecimenMessages({
  specimen,
  index,
}: {
  specimen: Specimen;
  index: number;
}) {
  return (
    <article
      className="specimen"
      aria-label={`Specimen ${index + 1}: ${specimen.agent_id}`}
    >
      <div className="specimen-label">
        <Badge variant="outline">Specimen {index + 1}</Badge>
        <span>Round {specimen.round_idx}</span>
        <span aria-hidden="true">/</span>
        <span>{humanize(specimen.task_kind)}</span>
        <span aria-hidden="true">/</span>
        <span>{specimen.task_id}</span>
      </div>
      <MessageGroup className="specimen-messages">
        <Message>
          <MessageAvatar className="message-avatar self-start rounded-lg">
            <Bot className="size-4" aria-hidden="true" />
          </MessageAvatar>
          <MessageContent>
            <MessageHeader className="flex-wrap gap-2">
              {specimen.model_assigned === "none"
                ? "Simulation context"
                : "Assistant response"}
              <Badge
                variant="secondary"
                className="h-auto max-w-full min-w-0 whitespace-normal wrap-anywhere"
              >
                {specimen.model_assigned}
              </Badge>
            </MessageHeader>
            <Bubble variant="muted" className="w-full max-w-none">
              <BubbleContent className="w-full rounded-lg">
                <p className="recorded-message">
                  {specimen.assistant_response_text ||
                    "No assistant response was recorded."}
                </p>
              </BubbleContent>
            </Bubble>
            <MessageFooter>Saved text · not a live reply</MessageFooter>
          </MessageContent>
        </Message>
        <Message>
          <MessageAvatar className="message-avatar self-start rounded-lg bg-background">
            <UserRound className="size-4" aria-hidden="true" />
          </MessageAvatar>
          <MessageContent>
            <MessageHeader className="flex-wrap gap-2">
              Simulated agent
              <Badge variant="outline">{specimen.agent_id}</Badge>
            </MessageHeader>
            <Bubble variant="outline" className="w-full max-w-none rounded-lg">
              <BubbleContent className="w-full rounded-lg">
                <p className="recorded-message">
                  {specimen.agent_response_text ||
                    "No agent response was recorded."}
                </p>
              </BubbleContent>
            </Bubble>
            <MessageFooter className="flex-wrap gap-x-4 gap-y-1">
              <span>Stance probability {format(specimen.stance_prob)}</span>
              <span>Confidence {format(specimen.confidence)}</span>
            </MessageFooter>
          </MessageContent>
        </Message>
      </MessageGroup>
      <div className="trace-record">
        <Trace>
          <TraceCollapsible>
            <TraceCollapsibleTrigger
              aria-label={`Tools and skills · specimen ${index + 1}`}
            >
              <TraceLine>
                <TraceIcon>
                  <Wrench />
                </TraceIcon>
                <TraceTitle>Tools & skills</TraceTitle>
                <Badge variant="outline">Not recorded</Badge>
              </TraceLine>
            </TraceCollapsibleTrigger>
            <TraceCollapsibleContent>
              <div className="prompt-availability">
                <h3>Tool calls</h3>
                <p>
                  This export does not include tool names, arguments, results,
                  or execution status.
                </p>
                <h3>Skill use</h3>
                <p>
                  This export does not include skill names, versions, or
                  invocation records.
                </p>
                <p>
                  Missing activity records do not establish that no tools or
                  skills were used. Extracted topics describe response content,
                  not execution history.
                </p>
              </div>
            </TraceCollapsibleContent>
          </TraceCollapsible>
          <TraceCollapsible>
            <TraceCollapsibleTrigger
              aria-label={`System prompts · specimen ${index + 1}`}
            >
              <TraceLine>
                <TraceIcon>
                  <FileText />
                </TraceIcon>
                <TraceTitle>System prompts</TraceTitle>
                <Badge variant="outline">Not recorded</Badge>
              </TraceLine>
            </TraceCollapsibleTrigger>
            <TraceCollapsibleContent>
              <div className="prompt-availability">
                <h3>Assistant system prompt</h3>
                <p>
                  {specimen.model_assigned === "none"
                    ? "Not applicable: this baseline has no AI assistant in the loop."
                    : "The exact assistant system prompt is not included in this saved trace."}
                </p>
                <h3>Simulated-agent system prompt</h3>
                <p>
                  The exact agent system prompt is not included in this saved
                  trace. Response text and model identity cannot establish which
                  instructions were used.
                </p>
              </div>
            </TraceCollapsibleContent>
          </TraceCollapsible>
          <TraceCollapsible>
            <TraceCollapsibleTrigger
              aria-label={`Recorded context · specimen ${index + 1}`}
            >
              <TraceLine>
                <TraceIcon>
                  <Fingerprint />
                </TraceIcon>
                <TraceTitle>Recorded context</TraceTitle>
              </TraceLine>
            </TraceCollapsibleTrigger>
            <TraceCollapsibleContent>
              <dl className="key-values">
                <div>
                  <dt>Model family</dt>
                  <dd>{specimen.model_family}</dd>
                </div>
                <div>
                  <dt>Agent region</dt>
                  <dd>{humanize(specimen.agent_region)}</dd>
                </div>
                <div>
                  <dt>Task</dt>
                  <dd>{specimen.task_id}</dd>
                </div>
              </dl>
            </TraceCollapsibleContent>
          </TraceCollapsible>
          <TraceCollapsible>
            <TraceCollapsibleTrigger
              aria-label={`Extracted topics · specimen ${index + 1}`}
            >
              <TraceLine>
                <TraceIcon>
                  <FileText />
                </TraceIcon>
                <TraceTitle>Extracted topics</TraceTitle>
                <Badge variant="secondary">
                  {specimen.extracted_topics.length}
                </Badge>
              </TraceLine>
            </TraceCollapsibleTrigger>
            <TraceCollapsibleContent>
              <div className="flex flex-wrap gap-2">
                {specimen.extracted_topics.map((topic, topicIndex) => (
                  <Badge
                    key={`${topic}-${topicIndex}`}
                    variant="outline"
                    className="h-auto whitespace-normal"
                  >
                    {humanize(topic)}
                  </Badge>
                ))}
              </div>
              {!specimen.extracted_topics.length && <p>No topics recorded.</p>}
            </TraceCollapsibleContent>
          </TraceCollapsible>
          <TraceCollapsible>
            <TraceCollapsibleTrigger
              aria-label={`Raw record · specimen ${index + 1}`}
            >
              <TraceLine>
                <TraceIcon>
                  <Braces />
                </TraceIcon>
                <TraceTitle>Raw record</TraceTitle>
              </TraceLine>
            </TraceCollapsibleTrigger>
            <TraceCollapsibleContent>
              <pre tabIndex={0} aria-label={`Raw JSON specimen ${index + 1}`}>
                <code>{JSON.stringify(specimen, null, 2)}</code>
              </pre>
            </TraceCollapsibleContent>
          </TraceCollapsible>
        </Trace>
      </div>
    </article>
  );
}

export function ResponseInspector({
  conditions,
  selected,
  specimens,
  onSelect,
  baselineConditions,
  onSelectBaseline,
}: {
  conditions: Condition[];
  selected?: Condition;
  specimens: Specimen[];
  onSelect: (id: string) => void;
  baselineConditions: Condition[];
  onSelectBaseline: (id: string) => void;
}) {
  const records = specimens.filter(
    (record) => record.condition_id === selected?.condition_id,
  );
  return (
    <div className="trace-layout">
      <aside
        className="surface trace-controls"
        aria-label="Response trace controls"
      >
        <div className="trace-condition-field">
          <label className="field-label" htmlFor="trace-condition">
            Trace condition
          </label>
          <select
            id="trace-condition"
            className="control"
            value={selected?.condition_id ?? ""}
            onChange={(event) => onSelect(event.target.value)}
          >
            {!conditions.length && (
              <option value="">No matching conditions</option>
            )}
            {conditions.map((condition) => (
              <option
                key={condition.condition_id}
                value={condition.condition_id}
              >
                {condition.label}
              </option>
            ))}
          </select>
        </div>
        <div className="trace-scope">
          <Badge variant="secondary">{records.length} specimens</Badge>
          <p>
            Sampled final-round records. The overview round filter does not
            apply.
          </p>
        </div>
        <details className="trace-explanation">
          <summary>About these records</summary>
          <p>
            These are saved response pairs from the Python report, not a
            reconstructed conversation. Context and topics are recorded fields,
            not private reasoning or tool calls.
          </p>
        </details>
      </aside>
      <details className="surface condition-setup" open>
        <summary>Condition setup & baselines</summary>
        <div className="setup-grid">
          {selected && (
            <div>
              <h3>Selected condition</h3>
              <p>{selected.label}</p>
              <dl className="key-values">
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
                <div>
                  <dt>Provenance</dt>
                  <dd>{humanize(selected.nationality_type)}</dd>
                </div>
                <div>
                  <dt>Regime</dt>
                  <dd>{humanize(selected.heterogeneity_type)}</dd>
                </div>
              </dl>
            </div>
          )}
          <div>
            <h3>Baseline conditions</h3>
            {baselineConditions.map((baseline) => (
              <div key={baseline.condition_id} className="baseline-setup">
                <p>{baseline.label}</p>
                <dl className="key-values">
                  <div>
                    <dt>Models</dt>
                    <dd>{baseline.num_models}</dd>
                  </div>
                  <div>
                    <dt>Temperature / top-p</dt>
                    <dd>
                      {baseline.temperature} / {baseline.top_p}
                    </dd>
                  </div>
                  <div>
                    <dt>Regime</dt>
                    <dd>{humanize(baseline.heterogeneity_type)}</dd>
                  </div>
                </dl>
                {baseline.heterogeneity_type === "no_ai" && (
                  <p className="panel-description">
                    No AI assistant; simulated agents exchange peer information.
                    Decoding settings are saved configuration, not evidence of
                    an assistant call.
                  </p>
                )}
                <Button
                  variant="outline"
                  className="mt-3"
                  disabled={selected?.condition_id === baseline.condition_id}
                  onClick={() => onSelectBaseline(baseline.condition_id)}
                >
                  {selected?.condition_id === baseline.condition_id
                    ? "Viewing baseline"
                    : "View baseline traces"}
                </Button>
              </div>
            ))}
            {!baselineConditions.length && (
              <p>No baseline conditions are included in this report.</p>
            )}
          </div>
        </div>
      </details>
      <section className="surface trace-conversation">
        <div className="panel-heading">
          <div>
            <h2>Recorded response specimens</h2>
          </div>
          <Badge variant="outline">Read-only</Badge>
        </div>
        {selected && <p className="panel-description">{selected.label}</p>}
        {records.map((specimen, index) => (
          <SpecimenMessages
            key={`${specimen.condition_id}-${specimen.agent_id}-${specimen.task_id}-${index}`}
            specimen={specimen}
            index={index}
          />
        ))}
        {!records.length && (
          <p className="empty-state">
            No response specimens are available for this selection.
          </p>
        )}
      </section>
    </div>
  );
}
