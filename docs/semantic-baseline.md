# Semantic NLU baseline

Dataset: `tests/data/semantic_queries.json` (120 cases).  
Parser: original deterministic `QueryParser`, measured before the semantic NLU implementation.

| Metric | Baseline |
|---|---:|
| Intent accuracy | 50.0% (60/120) |
| Ticker extraction accuracy | 70.8% (85/120) |
| Period extraction accuracy | 66.7% (4/6) |
| Follow-up resolution accuracy | 25.0% (6/24) |
| Clarification accuracy | 0.0% (0/5) |
| Multi-intent accuracy | 0.0% (0/5) |

Reproduce the evaluation with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_semantics.py --show-errors
```

The dataset and evaluator were created before replacing the legacy keyword-first parser. The baseline has no confidence field, so error rows report `confidence=None`.
