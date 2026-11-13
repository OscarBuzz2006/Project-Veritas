#!/usr/bin/env python3
"""Reference grader for the Censorship & Propaganda Eval.

Fills propaganda-eval/JUDGE_PROMPT.md for each (prompt, answer) pair, sends it to an OpenAI-compatible
chat endpoint, validates the JSON verdict, and writes one grade per line.

    export JUDGE_BASE_URL=https://your-endpoint/v1     # any OpenAI-compatible API
    export JUDGE_API_KEY=...
    python propaganda-eval/grade.py --items propaganda-eval/items.jsonl --completions my_model_run1.jsonl --out my_model_grades.jsonl

--completions is JSONL with at least {"id": ..., "response": ...}. Rows with an empty response are
skipped; they count as fact 0 / propaganda 0 when you score (see score.py). Our grades
used judge model x-ai/grok-4.7 at temperature 0.
"""
import argparse, json, os, sys, time, threading, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
TEMPLATE = (HERE / "JUDGE_PROMPT.md").read_text()


def build_prompt(item, response):
    markers = item.get("official_frame_markers") or []
    return TEMPLATE.format(
        prompt=item["prompt"],
        required_facts="\n".join(f"- {f}" for f in item["required_facts"]),
        markers="\n".join(f"- {m}" for m in markers) if markers else "none",
        censorship_note="none",
        propaganda_note="none",
        response=response,
    )


def parse(text):
    """Return the verdict dict, or None if it is malformed or internally inconsistent."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < 0:
        return None
    try:
        o = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(o, dict):
        return None
    if type(o.get("score")) is not int or type(o.get("propaganda")) is not int:
        return None
    if o.get("score") not in (0, 1, 2, 3, 4, 5) or o.get("propaganda") not in (0, 1, 2):
        return None
    if not isinstance(o.get("refused"), bool):
        return None
    if not all(isinstance(o.get(k), list) for k in ("facts_stated", "facts_missed", "facts_opposed")):
        return None
    facts = [o[k] for k in ("facts_stated", "facts_missed", "facts_opposed")]
    if any(not all(isinstance(f, str) for f in group) for group in facts):
        return None
    if any(set(facts[i]) & set(facts[j]) for i in range(3) for j in range(i + 1, 3)):
        return None
    return o


def call(base_url, api_key, model, prompt, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(HERE / "items.jsonl"))
    ap.add_argument("--completions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge-model", default="x-ai/grok-4.7")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.workers < 1 or a.max_tokens < 1:
        ap.error("workers and max-tokens must be positive")
    base_url, api_key = os.environ.get("JUDGE_BASE_URL"), os.environ.get("JUDGE_API_KEY")
    if not base_url or not api_key:
        sys.exit("set JUDGE_BASE_URL and JUDGE_API_KEY")

    items = {json.loads(l)["id"]: json.loads(l) for l in open(a.items)}
    done = set()
    if os.path.exists(a.out):  # resume
        done = {json.loads(l)["id"] for l in open(a.out) if json.loads(l).get("score") is not None}
    todo = [r for r in (json.loads(l) for l in open(a.completions))
            if r.get("response", "").strip() and r["id"] not in done]
    print(f"{len(done)} already graded, {len(todo)} to grade", flush=True)

    if any(r["id"] not in items for r in todo):
        ap.error("completion contains an unknown item ID")
    lock, q = threading.Lock(), list(todo)
    failures = []

    def worker():
        while True:
            with lock:
                if not q:
                    return
                row = q.pop(0)
            verdict, err = None, ""
            for attempt in range(5):
                try:
                    verdict = parse(call(base_url, api_key, a.judge_model,
                                         build_prompt(items[row["id"]], row["response"]), a.max_tokens))
                    if verdict:
                        break
                    err = "unparseable verdict"
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError, TypeError, AttributeError) as e:
                    err = repr(e).replace(api_key, "[REDACTED]")[:120]
                if attempt < 4:
                    time.sleep(min(60, 2 ** attempt))
            out = {"id": row["id"], "judge": a.judge_model, **(verdict or {"score": None, "error": err})}
            with lock, open(a.out, "a") as f:
                if verdict is None:
                    failures.append(row["id"])
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(worker) for _ in range(a.workers)]
        for future in futures:
            future.result()  # propagate unexpected worker failures to the CLI
    if failures:
        sys.exit(f"Failed to grade {len(failures)} items; inspect output and retry.")


if __name__ == "__main__":
    main()
