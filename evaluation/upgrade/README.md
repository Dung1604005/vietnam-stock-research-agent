# Upgrade evaluation fixtures

`benchmark-v1.json` is author-created, synthetic regression/calibration data. It is not an independent user holdout, not company disclosures and not market observations. The 320 NLU cases have 16 templates; do not count ticker substitutions as independent linguistic diversity. Conflict and screening cases also repeat scenario families.

The 20 RAG documents use reserved example.invalid URLs and educational content. They are loaded into an isolated temporary SQLite store below `.runtime/evaluations/`, never the production research database. The 30 single-stock dates are scenario metadata for routing checks, not historical price evidence.

Run `python scripts/evaluate_upgrade.py --strict` using the project venv. The JSON report includes dataset SHA-256, measured gates, unmeasured metrics, case counts and errors. LLM/citation precision with live providers and independently labelled company reports requires a separate evaluation set.
