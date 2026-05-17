# What "deep research" actually is: Building a multi-agent research graph in LangGraph

*A 12-agent system that classifies your question, plans an investigation, dispatches specialists in parallel, audits its own citations, and writes a refined cited markdown report. With code.*

I kept reading articles that promised to explain "deep research agents" and ended up describing a single LLM in a while-loop with a Tavily tool bound to it. That is not deep research. That is search with extra steps.

The product launches from OpenAI, Perplexity, Gemini and Anthropic over the last year have done something useful for the field, which is normalise the idea that one model alone is not going to brief you on a hard question. But the open-source posts I've seen lag behind. Most of them are wrappers around an LLM call, dressed up as something brand new (and of course I'm NOT going to claim I've built something wildly novel here either). What I did build is closer to what the term should mean: a graph of role-specialised agents that argue with each other, plan, fan out in parallel, then audit themselves before writing a single word of the final report.

I'll walk you through it.

For context, imagine you're a strategy analyst at Safaricom. Leadership wants a brief by Friday on how the major telcos in the region are deploying agentic AI in customer support. Where do you even start? A single chatbot will hallucinate half of it. A junior analyst will spend two days reading blog posts and still miss the regulator angle. What you actually want is something that thinks like a small research team: someone to scout the landscape, someone to plan the report, several people to read sources in parallel, someone to check the citations, and someone to write it up. That's the system.

The full code is on GitHub. As always, all code is open-sourced so you can clone it, point it at your own keys and run it on whatever question you're trying to brief leadership on.

## TLDR

A user question goes into a LangGraph state machine. A classifier picks one of four routes: a meta-question about the system itself, a clarification request, a shallow lookup, or a deep investigation. The deep path scouts the landscape, plans a structured outline, fans out across six role-specialised researchers, merges their notes, decides if there are gaps, audits the citation list, and only then drafts and refines the final report. The writer is structurally prevented from running before the citations have been audited. Press Ctrl-C halfway through and you still get a report from whatever was already collected.

## 1. Why a single LLM in a search loop is not enough

The naive version of a research agent is one prompt that says "you are a researcher, use this search tool, answer the question." It works for simple lookups. *"What year was LangGraph released?"* is fine. The problem is everything that isn't a one-hop fact.

Three things break the single-agent setup once the question gets harder:

- **No plan, no shape.** A single agent makes up the structure of the report on the fly. You get whatever order the model felt like writing in, which usually means executive-summary, then a wall of bullet points.
- **No specialisation.** A "skeptic" reads sources very differently from a "comparator" or a "horizon scanner." A single model trying to be all of them at once tends to be none of them.
- **No verification step.** The same model that picked the sources writes the report citing them. There is nobody between source-selection and writing whose job is to ask: *is this source actually any good, and does the claim in the draft match what the source says?*

This is the same reason real research teams have analysts, editors and fact-checkers. The roles exist because you need different kinds of attention applied at different points. The graph below is just that, expressed as code.

## 2. The graph

Here is the topology. It looks intimidating, but each node has one job.

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

Three things to notice on a first read.

The first one is that there are two distinct paths. A *shallow path* for cheap factual lookups, and a *deep path* for the real investigations. The classifier picks. If the shallow path picks up the question and then realises midway that it's harder than it looked, it can escalate to the deep path. This is important because you don't want a fusion-energy commercialisation question accidentally answered by a three-search ReAct loop.

The second is the parallel fan-out from the **architect**. The architect node produces a plan, and for each task in that plan it emits a LangGraph `Send`, which spawns a specialist node with its own ReAct loop. Six specialists can be reading sources at the same time. They merge their notes back into the global state through a None-safe list reducer, so two specialists writing notes concurrently never overwrite each other.

The third is the loop between `gap_analyzer` and the specialists. After synthesis, the gap analyser reads the merged brief and decides whether the report is ready. If not, it emits new `Send`s with follow-up tasks. This keeps going until either the gap analyser says "sufficient" or we hit the iteration cap. (More on the cap in a moment, it's enforced in code, not in the model's good intentions.)

## 3. The 12 agents and what each one does

Below is the full cast. Two of them, `meta` and `clarifier`, are short-circuit branches. The rest do the actual work.

| Agent              | Role                                                                 |
|--------------------|----------------------------------------------------------------------|
| classifier         | Decides the route: meta / shallow / deep / clarification.            |
| meta               | Answers questions about the system itself ("what can you do?").      |
| clarifier          | Asks the single most important clarifying question, then exits.     |
| shallow_researcher | A lightweight ReAct loop for simple factual lookups. Cheap tier.     |
| scout              | A broad first pass that maps the research landscape before planning. |
| architect          | Builds the outline and assigns roles to each specialist.             |
| evidence_gatherer  | Specialist. Finds primary sources, official data, named numbers.     |
| mechanism_explorer | Specialist. Explains *how* and *why* things work.                    |
| comparator         | Specialist. Side-by-side comparisons, trade-offs, benchmarks.        |
| skeptic            | Specialist. Actively hunts for counter-evidence and weak claims.     |
| horizon_scanner    | Specialist. Reads ahead: trends, forecasts, what's likely next.     |
| domain_specialist  | Specialist. Brings the domain frame (regulator, clinician, etc.).   |
| synthesis          | Merges all specialist notes into a structured brief on the outline.  |
| gap_analyzer       | Decides sufficient/insufficient; emits follow-up tasks if needed.    |
| citation_audit     | Dedups sources, builds a claim → source map, flags weak citations.  |
| writer             | Drafts the report, citing ONLY audited sources.                      |
| refiner            | Polishes the draft without changing evidence or citations.           |

The point of the cast list is that the *skeptic* and the *evidence gatherer* are reading the same web. They are looking for different things because their prompts tell them to. The skeptic is the closest thing the system has to an internal critic, and on a topic like agentic-AI hype its notes are usually the most useful section in the synthesis.

For those who may not have heard of this term, a **ReAct loop** is just an LLM in a small loop where, on each step, the model can either think out loud, call a tool (here it's Tavily search or extract), or decide it's done. Each specialist gets its own ReAct loop with its own recursion cap. Think of it as a junior analyst with a stack of browser tabs and a deadline.

## 4. Core components of the deep path

I'll break the deep path down further, because this is where most of the design work went.

### a. The scout and the architect

The scout is doing reconnaissance. It does *not* try to answer the question. It tries to map the territory: what are the major sub-topics, who are the players, what's controversial, what's settled. Its output is a research landscape that the architect then turns into a plan.

The architect is the most opinionated agent in the graph. It picks which specialist roles are even relevant for this particular question. A pure technical comparison ("LangGraph vs CrewAI vs AutoGen") doesn't need a *domain_specialist*. A regulatory question about M-PESA agent fraud absolutely does. The architect skips the roles that won't earn their tokens.

### b. The specialists and the synthesis

Each specialist gets two Tavily tools bound to it. `tavily_search` returns title plus URL plus a short snippet, the way a web search does. `tavily_extract` is the important one, it returns the full clean text of specific URLs. The prompt tells the specialist to use Search to find candidates, then Extract to actually read the top one to three picks before citing. This means citations end up containing verbatim passages from the page rather than search-result snippets. Fewer hallucinations, better quotes in the final report, at the cost of about a thousandth of a dollar per extracted URL.

Once the specialists finish, the synthesis node merges their notes into a structured brief mapped to the architect's outline. Synthesis is on the cheap tier on purpose. It's a merging job, not a thinking job.

### c. The gap analyzer

This is the loop. The gap analyser reads the synthesised brief and decides whether the report is ready to write. If it says "insufficient", it emits new follow-up `ResearchTask`s, which become `Send`s to the specialists, which run another parallel round. The loop keeps going until either the gap analyser is satisfied or we hit `DEEP_RESEARCH_MAX_ITER` (default 2).

That iteration cap is enforced in routing code, not in the model's prompt. The model can say "insufficient" all day, but the router moves on regardless once `iterations >= MAX_DEEP_ITERATIONS`. This is a small thing that makes a big difference, because the alternative is paying for an infinite loop the first time the gap analyser develops a perfectionist streak.

### d. Citation audit, writer, refiner

These three are the production line at the end. The citation audit dedups sources across all specialist notes, classifies each source as high/medium/low quality, and emits a claim → source registry that becomes the writer's allowed-citations list. The writer drafts the report citing *only* sources from that registry. The refiner gets the draft (and nothing else) and polishes it for tone and flow.

The refiner does not see the evidence. That's deliberate. A polishing model that can see the sources can be tempted to "improve" the citations. A polishing model that only sees the draft can only improve the draft.

## 5. The safety properties (the part I'm most proud of)

This is the part that distinguishes the build from a single-agent search loop. Four properties are enforced by the graph topology and the code, not by trust in the model:

1. **The writer is unreachable except via citation_audit.** There is only one edge into the writer node, and it comes from the auditor. This is checked by a unit test, `tests/test_graph_build.py::test_writer_only_runs_after_citation_audit`. If anyone (including future me) accidentally adds a `synthesis → writer` edge, the test fails.
2. **The writer cites only audited sources.** The writer's prompt is given the registry from `citation_audit` and instructed to use only those IDs. Combined with property 1, you cannot have a final report with un-audited citations in it.
3. **The refiner cannot introduce new facts or citations.** Its prompt enforces this, and it is structurally given only the draft, not the underlying evidence. The refiner is for tone, not for content.
4. **The iteration cap is in code, not in the model's judgement.** The routing function moves to `citation_audit` once iterations hit the cap, regardless of what `gap_analyzer` said.

The relevant routing snippet, lightly abridged:

```python
def _route_after_gap(state: ResearchState) -> list[Send] | str:
    iterations = state.get("iterations", 0)
    gap = state.get("gap_analysis")

    # Cap is enforced in code, not in the model's judgement.
    if iterations >= MAX_DEEP_ITERATIONS:
        return "citation_audit"

    if gap and gap.status == "insufficient" and gap.follow_ups:
        return [Send("specialist", task) for task in gap.follow_ups]

    return "citation_audit"
```

That single `if iterations >= MAX_DEEP_ITERATIONS` line is the difference between "deep research agent" and "deep research bill."

## 6. Press Ctrl-C and still get a report

One of my favourite parts to build was the early-exit behaviour. A deep research run can take a few minutes (15+ search queries, multiple heavy-tier completions). When you're iterating on prompts, that's painful. So the CLI has two off-ramps: a `--max-time` budget, and a Ctrl-C handler. Both end up calling the same helper, `finalize_from_partial(state)` in `src/deep_research/finalize.py`.

`finalize_from_partial` walks downstream from wherever the partial state stopped and produces the cheapest final report it can. The branching looks like this:

| State at exit                       | What finalize does                                            |
|-------------------------------------|----------------------------------------------------------------|
| `final_report` already set          | Return it.                                                     |
| `draft_report` present              | Refiner only.                                                  |
| `specialist_notes` present          | (Bootstrap a plan if missing) → synthesis → audit → write → refine. |
| `shallow_answer` present            | Audit (if missing) → shallow_finalize.                         |
| Nothing useful collected            | Short "Interrupted" placeholder noting how far we got.         |

The caveat (and I want to be honest about this one) is that the wall-clock check runs *between* nodes. A single long specialist call can overshoot by tens of seconds. And work inside an in-flight node is lost on Ctrl-C, only the outputs of completed nodes are in the partial state. But for the common case, which is "I waited two minutes and I have what I need", you press Ctrl-C and you still get a cited report instead of nothing.

## 7. Provider flexibility (use what you have)

I built this to work with whichever LLM provider you happen to have keys for. The provider is resolved at the first model call, in this order:

1. If `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` is set, that wins.
2. Otherwise it auto-detects from keys. OpenAI if `OPENAI_API_KEY` is set, Anthropic if only `ANTHROPIC_API_KEY` is set, OpenAI if both are set.
3. Either way, you can override any single role with a `<ROLE>_MODEL` env var, and the provider is inferred from the model-name prefix. So you can mix.

That last point is the practical one. The writer is the most quality-sensitive role in the system, and on the deep path it benefits from running on a heavier model. The synthesis node is just merging text, so it can run on a cheap one. A sensible default if you have both keys: `WRITER_MODEL=claude-opus-4-7 SPECIALIST_MODEL=gpt-4.1 SYNTHESIS_MODEL=gpt-4.1-mini`. Mix and match. Cheap where you can, heavy where you must.

FYI for cost-watchers: pushing every heavy role down to the cheap tier (`WRITER_MODEL=gpt-4.1-mini SPECIALIST_MODEL=gpt-4.1-mini`, or the Anthropic equivalent `WRITER_MODEL=claude-sonnet-4-6 SPECIALIST_MODEL=claude-haiku-4-5-20251001`) cuts a deep-research run dramatically. The report quality drops, but for a lot of internal use cases that trade is the right one.

## 8. Running it

Enough of the theory. Let's jump to the interesting bits.

```bash
# CLI
deep-research "What is the state of fusion energy commercialization in 2026?"
deep-research "Compare LangGraph vs CrewAI vs AutoGen" --out report.md

# Force a path, bypassing the classifier
deep-research "What year was LangGraph released?" --path shallow
deep-research "Compare A vs B vs C"               --path deep

# Early exit, get a report from partial state instead of waiting it out
deep-research "..." --max-time 180   # soft 3-minute budget
deep-research "..." --max-iter 1     # one architect→specialist→synthesis cycle, no follow-ups

# Or just press Ctrl-C, the CLI catches it and finalizes what's in state.
```

For the Safaricom-analyst scenario at the top of the article, the command is just:

```bash
deep-research "How are major African telcos deploying agentic AI in customer support in 2025-2026?" --out brief.md
```

Two minutes later you have a markdown brief, cited, with claims you can trace back to the source URL. That's the artifact.

For the visual learners, the graph runs inside LangGraph Studio too:

```bash
pip install langgraph-cli
langgraph dev
```

This is genuinely useful when debugging which specialist's notes a particular claim ended up in.

## 9. Known limitations

I want to be transparent about what this is and isn't.

- **Cost.** A single deep-research run can issue 15+ search queries and several heavy completions. Default settings are not free. Push roles down to the cheap tier (see above) if you're iterating.
- **Citation audit is best-effort.** The auditor classifies source quality from URL signals alone, it doesn't fetch and verify the pages. Treat `quality=high` as "looks authoritative", not "verified."
- **No persistence by default.** If you need resumability for long runs, drop a `SqliteSaver` checkpointer into `graph.py`. Easy change, I just didn't make it the default.
- **No paywalled content.** Tavily returns publicly accessible pages only. For paywalled sources you'd need to swap in a different fetcher.
- **Iteration cap is global.** If only one section of the report has gaps, all specialists still run their follow-ups in parallel. Could be smarter (this is on the list for a Part 2).

## 10. Summary and what's next

What I built is a multi-agent LangGraph deep research system that classifies the question, plans the investigation, dispatches role-specialised researchers in parallel, audits its own citations, and writes a refined cited report. The safety properties that matter (writer-after-audit, refiner-no-new-facts, code-enforced iteration cap, parallel-safe reducers) are enforced by the graph topology and tested, not left to the model's discretion. Pressing Ctrl-C still gives you a report.

For Part 2 I'd like to write up the targeted follow-up routing (only re-run specialists whose sections have gaps), checkpointer-backed resumability, and a proper evaluation pass on a small set of canned questions. If you want to see that one, leave a comment and tell me which part you're most interested in.

To be sincere, the most useful thing about building this was watching where the cheap-tier roles started failing. A `synthesis` node that's "just merging text" turns out to handle conflicting evidence very differently on the cheap tier versus the heavy one. There's a whole article in that observation alone.

---

The full code, as always, can be accessed [here](https://github.com/wandabwa2004/deep_research_agent). Clone the repo, change to the `deep_research_agent` folder, copy `.env.example` to `.env` and add either an `OPENAI_API_KEY` or an `ANTHROPIC_API_KEY` (plus a `TAVILY_API_KEY`), then run `pip install -e ".[dev]"` and you're ready to point it at a real question.

Don't forget to follow me, clap for me, and leave a comment. I'm also happy to connect via LinkedIn.

### Related articles

- *What Agentic AI Actually Is: A SIM Replacement Use Case (Part 1)*
- *Part 2: From Theory to Code, Building a SIM Swap Agent in LangGraph*
- *Building a RAG System with MMR for Safaricom's Smart Assistant*
- *Creating a Personalized Safari Guide with Agentic AI and Text-to-Speech (TTS)*
