# Deep Research Agent

A multi-agent deep research system built on **LangGraph**, inspired by NVIDIA
AIQ / AgentIQ deep-research patterns. Ask it a question — it classifies the
intent, optionally clarifies, scouts the landscape, plans a structured
investigation, dispatches role-specialised researchers in parallel, audits its
own citations, and writes a refined cited markdown report.

## Architecture

```
                                 ┌── meta ─────────────────┐
                                 │                         │
                                 ├── clarifier ────────────┤
START → classifier ─ROUTE ─────┤                         ├──→ END
                                 │ shallow_researcher      │
                                 │   ├─ sufficient → citation_audit → shallow_finalize → END
                                 │   └─ insufficient (escalate) ─┐
                                 │                                │
                                 └──→ scout ◀─────────────────────┘
                                       │
                                       ▼
                                  architect ─Send()─▶ specialist × N
                                       ▲                    │
                                       │                    ▼
                            (Send follow-ups)         synthesis
                                       │                    │
                                       │                    ▼
                                       └────── gap_analyzer
                                                            │ sufficient or max iter
                                                            ▼
                                                     citation_audit → writer → refiner → END
```

| Agent             | Role                                                       | OpenAI default | Anthropic default          |
|-------------------|------------------------------------------------------------|----------------|----------------------------|
| classifier        | Decides route: meta / shallow / deep / clarification       | gpt-4.1-mini   | claude-haiku-4-5-20251001  |
| meta              | Answers questions about the system itself                  | gpt-4.1-mini   | claude-haiku-4-5-20251001  |
| clarifier         | Asks the single most important clarifying question         | gpt-4.1-mini   | claude-haiku-4-5-20251001  |
| shallow_researcher| Lightweight ReAct loop for simple factual lookups          | gpt-4.1-mini   | claude-haiku-4-5-20251001  |
| scout             | Broad exploratory pass → research landscape                | gpt-4.1        | claude-sonnet-4-6          |
| architect         | Builds plan + outline + assigns specialist roles           | gpt-4.1        | claude-sonnet-4-6          |
| specialist × 6    | evidence_gatherer, mechanism_explorer, comparator,         | gpt-4.1        | claude-sonnet-4-6          |
|                   | skeptic, horizon_scanner, domain_specialist                |                |                            |
| synthesis         | Merges notes → structured brief mapped to outline          | gpt-4.1-mini   | claude-haiku-4-5-20251001  |
| gap_analyzer      | Decides sufficient / emits follow-up tasks                 | gpt-4.1        | claude-sonnet-4-6          |
| citation_audit    | Dedup sources, claim→source map, flag weak citations       | gpt-4.1        | claude-sonnet-4-6          |
| writer            | Draft report citing ONLY audited sources                   | gpt-4.1        | claude-opus-4-7            |
| refiner           | Polish without changing evidence or citations              | gpt-4.1-mini   | claude-haiku-4-5-20251001  |

**Provider selection** (resolved at first model call):

1. `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` env var wins if set.
2. Otherwise auto-detect from keys: OpenAI if `OPENAI_API_KEY` is set,
   Anthropic if only `ANTHROPIC_API_KEY` is set, OpenAI if both are set.
3. Either way, you can still override any single role via `<ROLE>_MODEL` —
   provider is inferred from the model-name prefix, so you can mix
   (e.g. `WRITER_MODEL=claude-opus-4-7` even when the default provider is OpenAI).

### Safety properties enforced by the graph

- **Writer never runs before citation_audit.** The graph topology has only one
  edge into `writer`: `citation_audit → writer`. Verified by
  `tests/test_graph_build.py::test_writer_only_runs_after_citation_audit`.
- **Writer cites only audited sources.** The writer is prompted with the
  registry from `citation_audit` and instructed to use only those ids.
- **Refiner cannot introduce new facts or citations.** Prompt enforces this;
  the refiner gets only the draft, not the raw evidence.
- **Iteration cap is enforced in code, not in the model's judgement.**
  `gap_analyzer` may say "insufficient", but the routing function still moves
  to `citation_audit` once `iterations >= DEEP_RESEARCH_MAX_ITER`.
- **Parallel notes accumulate safely** via a None-safe list reducer on
  `specialist_notes`, `shallow_citations`, `follow_up_tasks`, and `warnings`.

## Setup

```bash
cd ~/Projects/deep_research_agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in OPENAI_API_KEY OR ANTHROPIC_API_KEY, plus TAVILY_API_KEY
```

## Run

```bash
# CLI
deep-research "What is the state of fusion energy commercialization in 2026?"
deep-research "Compare LangGraph vs CrewAI vs AutoGen" --out report.md

# Force a path, bypassing the classifier
deep-research "What year was LangGraph released?" --path shallow
deep-research "Compare A vs B vs C"            --path deep

# Early exit — get a report from partial state instead of waiting it out
deep-research "..." --max-time 180        # soft 3-minute budget
deep-research "..." --max-iter 1          # one architect→specialist→synthesis cycle, no follow-ups
# Or press Ctrl-C at any time — the CLI catches it and finalizes what's in state.

# LangGraph Studio
pip install langgraph-cli
langgraph dev
```

### How `--path` works

Without the flag, the classifier picks `meta` / `shallow_research` /
`deep_research` / `needs_clarification` from the question text. With
`--path shallow|deep`:

- The CLI pre-populates `state["intent"]`, so `classifier_node` short-circuits
  with zero LLM calls.
- It also sets `state["force_path"]`, which suppresses the shallow path's
  auto-escalation. So `--path shallow` always finishes on the shallow path,
  even if the shallow researcher self-judges as `INSUFFICIENT`.
- `meta_response` and `needs_clarification` are not exposed as flag values —
  they're classifier-only outcomes.

### How early exit works

When `--max-time` is exceeded *between* nodes, or you press Ctrl-C, the CLI
runs `finalize_from_partial(state)` (`src/deep_research/finalize.py`). That
helper picks the cheapest path that still produces a report from what's
already in state:

| State at exit                       | What finalize does                                  |
|-------------------------------------|------------------------------------------------------|
| `final_report` already set          | return it                                            |
| `draft_report` present              | refiner only                                         |
| `specialist_notes` present          | (bootstrap plan if missing) → synthesis → audit → writer → refiner |
| `shallow_answer` present            | audit → shallow_finalize                             |
| Nothing useful collected            | short "Interrupted" placeholder noting how far we got |

Caveats:
- The wall-clock check runs *between* nodes, so a single long specialist
  could overshoot by tens of seconds.
- Work inside an in-flight node (e.g. an LLM call that hasn't returned) is
  lost on Ctrl-C — only completed nodes' outputs are in the partial state.
- If you set `--max-iter 1`, the gap analyzer's follow-up loop is skipped
  entirely (faster, less thorough).

## Tests

```bash
pytest                     # unit + routing + graph-topology tests (no API keys)
python scripts/smoke.py    # end-to-end against three canned questions (requires keys)
```

The pytest suite covers:
- routing for all four classifier branches
- shallow → audit vs. shallow → escalate-to-scout
- architect fan-out emits one Send per task
- gap_analyzer triggers follow-up Sends, respects max iterations
- writer/refiner state-flow with a fake model
- graph topology: writer only reachable from citation_audit, refiner only from writer
- list reducer is None-safe (parallel branches accumulate without overwriting)

## Configuration

Every agent's model is configurable via `<ROLE>_MODEL` env vars (see
`.env.example`). Provider is inferred from the model-name prefix:
- `claude*` → Anthropic
- `gpt*` / `o1*` / `o3*` → OpenAI

The system picks defaults from whichever provider has an API key set — see
the Provider selection rules above.

Runtime limits:
- `DEEP_RESEARCH_MAX_ITER` (default 2) — caps the architect→specialist→synthesis
  →gap_analyzer loop
- `SHALLOW_MAX_SEARCHES` (default 3) — caps searches for the shallow path
- `SPECIALIST_RECURSION_LIMIT` (default 18) — caps ReAct steps per specialist,
  covering both Tavily `search` and `extract` tool calls

## Tools the specialists use

Each specialist has two Tavily tools bound to it:

- `tavily_search` — discovery: returns title + URL + short snippet
- `tavily_extract` — reading: returns the full clean text of specific URLs

The prompt tells the specialist to use Search to find candidate sources, then
Extract to read the top 1–3 picks before citing. This means citations contain
verbatim passages from the page rather than search-result snippets — fewer
hallucinations and better quotes in the final report, at the cost of ~$0.001
per extracted URL.

## What changed from the previous version

Breaking changes from the linear `planner → researcher → summarizer → critic →
writer` chain:

- Old `Note`, `SubQuestion`, `Critique`, `ResearcherState` are gone. The
  equivalents are `SpecialistNote`, `ResearchTask`, `GapAnalysis`,
  `SpecialistState` — see `state.py`.
- Old hard-coded model names in `agents.py` are gone. Use `config.model_for(role)`.
- The final output field renamed: `state["report"]` → `state["final_report"]`.
- `agents.py` was split into `agents/` package — one module per role.
- The CLI surface is unchanged (`deep-research "..."`).

## Known limitations

- **Cost.** A single deep-research run can issue 15+ search queries and multiple
  heavy-tier completions. To cut cost, push heavy roles down to the cheap tier
  (e.g. `WRITER_MODEL=gpt-4.1-mini SPECIALIST_MODEL=gpt-4.1-mini`, or the
  Anthropic equivalent `WRITER_MODEL=claude-sonnet-4-6 SPECIALIST_MODEL=claude-haiku-4-5-20251001`).
- **Citation audit is best-effort.** The auditor classifies source quality from
  URL signals alone; it doesn't fetch and verify pages. Treat `quality=high`
  as "looks authoritative" not "verified".
- **No persistence by default.** Add a `SqliteSaver` checkpointer in
  `graph.py` if you need resumability for long runs.
- **No paywalled content.** Tavily returns publicly accessible pages only.
- **Iteration cap is global.** If only one section of the report has gaps, all
  specialists run their follow-ups concurrently. Could be smarter.
