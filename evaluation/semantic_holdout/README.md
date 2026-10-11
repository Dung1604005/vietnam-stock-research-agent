# Semantic holdout dataset

This folder is intentionally empty of labelled examples. A real holdout must be collected from anonymized user queries or authored by annotators who do not modify `intent_examples.py`, scoring weights, or the regression dataset.

`synthetic-v1.json` is a clearly marked synthetic smoke dataset requested for local testing. It is not independent user traffic and must not be used to declare calibration or release readiness.

Use an object with a version and a `cases` array. Each case uses the same fields as `tests/data/semantic_queries.json` (`query`, `expected_intent`, optional expected ticker/context/period fields).

Run it with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_semantics.py --holdout evaluation\semantic_holdout\your-version.json --write-report
```

The evaluator rejects normalized query overlap with the regression dataset. Do not copy regression cases here and do not label synthetic examples as user traffic.
