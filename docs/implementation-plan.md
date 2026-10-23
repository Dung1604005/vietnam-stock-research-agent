# Implementation baseline and remaining work

Updated 23/10/2026 against the working tree. The original five-role MVP plan has
been superseded by the implemented six-role architecture, including Research.
Historical phase plans remain linked in [the document register](README.md).

## Implemented baseline

- Six classes: Data, Research, Technical, Fundamental, Risk and Manager, each with AgentSpec and execution report.
- Dependency-injected providers/tools, Pydantic contexts/results, typed messages and role-scoped output validation.
- Full LangGraph with forecast/research, parallel specialists, Risk join, three re-analysis paths, optional evidence review/follow-up and deterministic finalization.
- Hybrid Vietnamese NLU, validated executable task plans, selective routes, comparison, bounded discovery/recommendation and charts.
- Market provenance/cache/raw snapshots/quality gates; research vector+FTS, HTML/text-layer PDF, versioned documents and policy-controlled crawler ingestion.
- KnowledgeService and graph projections; optional structured LLM tasks with evidence and numeric guards.
- STANDARD/QUICK/DEEP/OFFLINE budgets, partial results, local checkpoints/resume, SQLite conversations, replay and sanitized exports.

The graph and route details are maintained in [architecture.md](architecture.md);
the current status and evidence are in [project-report.md](project-report.md).

## Code responsibilities

| Location under src/stock_multi_agent | Responsibility |
|---|---|
| agents/ | Six bounded role policies; numeric scoring belongs here/Python helpers |
| graph/ | Full-analysis orchestration and bounded cycles |
| models/ | Pydantic contracts, state, messages, evidence, budgets |
| providers/ | Real vnstock integration, LLM adapters, resilience, offline replay |
| tools/ | Market-data wrapper and shared indicator/financial capabilities |
| services/ | Application APIs, execution, persistence, retrieval, forecasts, charts, evaluations |
| chat/, nlu/, ui/ | Routing/context/language interpretation, response and rendering |

`FinancialToolkit` supplies completeness/format helpers; financial scoring and
canonical facts are implemented in FundamentalAgent/services, not exclusively in
the toolkit. vnstock is a core dependency in pyproject.toml; OpenAI, TensorFlow,
pypdf and sentence-transformers are optional extras. Test-only providers and
explicit private replay have different purposes from live-data fallback.

## Remaining acceptance work

1. Independently review real issuer facts, periods/units, retrieval relevance and claim support.
2. Validate publication-time and adjusted-price/corporate-action history before point-in-time performance claims.
3. Measure live upstream/schema reliability and latency on the actual demo machine.
4. Complete forward validation/model gates; do not promote an artifact from software tests alone.
5. Add production authentication, user isolation, retention/migration/backup and concurrency evidence before multi-user use.
6. Keep autonomous background goals/workers separate: [autonomous-agents-plan.md](autonomous-agents-plan.md) is a proposal, not an implemented agent scheduler.

Project dependencies, caches and test temporary files stay inside the D-drive workspace.
