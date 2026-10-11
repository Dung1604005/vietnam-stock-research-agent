# Conversational assistant architecture

The original keyword-first `QueryParser` is now a compatibility facade over the hybrid semantic NLU documented in [semantic-nlu.md](semantic-nlu.md). Its output remains a typed `ParsedQuery`, extended with confidence, ambiguities, secondary intents, context resolution, clarification and privacy-minimal parser telemetry, so the five-agent analysis graph remains unchanged.

## Request lifecycle

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit Chat
    participant CS as ChatService
    participant Q as QueryParser
    participant R as IntentRouter
    participant C as ConversationState
    participant A as AnalysisService
    participant G as LangGraph / selected agents
    participant RC as ResponseComposer
    participant CH as ChartService
    participant K as KnowledgeService / SQLite FTS5
    User->>UI: Natural-language message
    UI->>CS: handle(message, conversation)
    CS->>Q: parse message + bounded context
    Q-->>CS: ParsedQuery
    CS->>R: plan(parsed query)
    R-->>CS: validated bounded RoutingPlan
    alt THEORY / CURRENT_RULE
        CS->>K: deterministic retrieve
        K-->>CS: chunk evidence + source metadata
    else MIXED
        CS->>K: retrieve theory evidence
        K-->>CS: chunk evidence + source metadata
        CS->>A: fetch/apply real ticker data
        A->>G: selected deterministic specialist
        G-->>CS: typed result + provenance
    else analysis / market route
        CS->>A: invoke route
        A->>G: full graph or selected agents
        G-->>CS: typed AgentState + trace
    end
    CS->>RC: compose user-facing response
    opt chart requested
        CS->>CH: safe ChartRequest + MarketData
        CH-->>CS: deterministic ChartArtifact
    end
    CS->>C: update ticker/intent/horizon/list memory
    CS-->>UI: AssistantResponse
```

## Typed contracts

- `ParsedQuery`: intent, tickers, market, sector, horizon, chart period, requested indicators, freshness requirement and follow-up marker.
- `RoutingPlan`: resolved intent plus allowlisted typed tasks with dependencies, input/output contracts, timeout and bounded budget. It is validated before dispatch, not a generic DAG executor.
- `ConversationState`: a bounded window of messages plus compact context fields. It stores at most 24 messages by default and summarizes evicted messages.
- `AssistantResponse`: display message, response type, tables, charts, trace, sources, warnings, freshness and UI metadata.
- `ChartRequest`: ticker, chart type, period and indicator names. It cannot contain executable code.
- `ChartArtifact`: normalized chart records produced by deterministic application code.
- `KnowledgeSearchResult`: citations linked to document/chunk, retrieval mode, score and freshness behavior.

## Knowledge questions

The semantic layer distinguishes `THEORY`, `CURRENT_RULE` and `MIXED`. Pure theory routes only to `KNOWLEDGE`, so it does not spend a vnstock request. Current-rule answers show publisher, URL, published/effective/retrieved timestamps and a freshness warning. MIXED requests such as `ROE của FPT có tốt không?` retrieve the definition from the local KB and obtain the actual metric from the existing Fundamental service; evidence origins are not merged into one unlabeled claim. See [knowledge-base.md](knowledge-base.md).

## Context resolution

The deterministic bilingual parser first extracts explicit entities. It then resolves omitted entities from memory:

- `Phân tích FPT` sets active ticker `FPT`.
- `Cho tôi biểu đồ 6 tháng` inherits `FPT` and sets chart period `6M`.
- `RSI thế nào?` inherits `FPT` and routes only Data → Technical with the RSI tool.
- `So với CMG?` combines active `FPT` with explicit `CMG`.
- `mã đầu tiên` resolves against the previous discovery result.

An LLM can later augment parsing, but deterministic parsing remains the safe fallback and owns no chart code or arbitrary execution.

## Discovery scope

Discovery reads `config/watchlist.json`. Results always disclose the number and category of configured symbols being screened; the system never presents this fixed set as the whole market. Failed symbols are reported as warnings instead of receiving invented values.

## UI rendering

The Streamlit page delegates to reusable render functions for user/assistant messages, analysis cards, overview metrics, stock/comparison tables, charts, agent status, expandable activity and the disclaimer. Plotly receives only `ChartArtifact` data produced by `ChartService`.
