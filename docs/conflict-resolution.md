# Conflict resolution between agents

The analysis graph uses bounded, evidence-driven coordination. Agents do not debate freely and the LLM does not decide which specialist wins.

## Conflict classes

- `SPECIALIST_DISAGREEMENT`: Technical and Fundamental return different non-UNKNOWN signals.
- `TECHNICAL_EVIDENCE_INSUFFICIENT`: Technical cannot produce a supported signal.
- `FUNDAMENTAL_EVIDENCE_INSUFFICIENT`: Fundamental lacks usable financial evidence.
- `HIGH_MEASURED_RISK`: Risk measures high issuer or market risk.
- `HIGH_DATA_UNCERTAINTY`: Risk finds the input evidence unreliable.
- `DATA_QUALITY_LIMITATION`: a potentially recoverable price or financial-data quality problem exists.
- `RISK_VETO`: a hard safety constraint applies; re-analysis stops unless a recoverable data problem could invalidate the veto evidence.

## Re-analysis actions

Manager selects one typed action rather than always rerunning Technical:

1. `DATA_PIPELINE`: bypass the ticker cache once, refresh market/financial data, refresh the forecast and research, then rerun Technical, Fundamental and Risk.
2. `RESEARCH_FUNDAMENTAL`: broaden citation retrieval, then rerun Fundamental and Risk.
3. `TECHNICAL`: use extended technical context, then rerun Risk.

The action is chosen from conflict type and specialist confidence. Data recovery has priority, followed by Fundamental evidence and Technical evidence. A `DATA_PIPELINE` attempt counts as having rerun all three branches.

## Loop controls

`max_iterations` is the maximum number of evidence-producing re-analysis rounds, not the number of mandatory debates. The default is two.

Manager stores the attempted action history and a deterministic evidence fingerprint. It stops when:

- the iteration budget is exhausted;
- the previous action did not change market, research, Technical or Fundamental evidence;
- every relevant action has already been attempted; or
- Risk issued a veto and no recoverable data-quality problem exists.

On stop, the existing deterministic finalization policy lowers confidence and returns a conservative result. Trace events expose `DETECT_CONFLICT`, `REVISE_PLAN`, `REQUEST_REANALYSIS`, and `STOP_LOOP` for audit and classroom demonstration.
