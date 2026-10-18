# LSTM V3: engineering pipeline and honest model status

Status: **USER_TRAINED_ARTIFACT_RECEIVED — AUDITED — NOT_PROMOTED**.
V2 is still the default. The user's full-training ZIP has been received, but
neither V3 horizon has passed the complete promotion gate.

Update 18/10/2026: the user has now supplied a full Kaggle training ZIP. It was
read-only audited, without replacing V2: [numerical gate correction and results](lstm-v3-numerical-gate-review.md).
The one-session development gate fails (1/3 winning folds); five-session passes
the development step (3/3), but neither horizon is policy eligible. Prior to this
submission, the implementation milestone was NOT_TRAINED; that is no longer the
status of the supplied full-training artifact. No model was promoted.

Preparation verified locally on 18/10/2026: the **new** CSV is at
`.runtime/ml/v3/dataset-1/vci_ohlcv_lstm_v3.csv` (524,950,632 bytes).
All 1,000,000 raw rows were retained, with 991 tickers, 8,255 conservative
segments and 333,690 eligible 30-step endpoints after feature warm-up,
liquidity/context checks and both labels. No row was removed because its future
return was large. Structural quarantine count is zero; this does **not** certify
corporate-action correctness. Unit/adjustment remain UNKNOWN for all rows.

## What stays unchanged

V2 data, notebooks and `notebooks/trainResult` remain frozen. Runtime V2 currently
contributes at most 20% to Technical under the existing demo policy, despite its
old manifest describing display-only use. This sidecar records the distinction;
it does not rewrite historical hashes or pretend V2 passed the new V3 gate.

## V3a specification

- Target: close-to-close raw return after 1 and 5 observed source sessions.
- Forecast origin: after the source closing bar is available, not before close.
- 17 single-ticker causal features, 30-row finite contiguous windows. No market
  median, cross-sectional percentile or frozen-mean cross-sectional proxy.
- Builder preserves source/fetched timestamp, unknown adjustment/unit, duplicate
  reconciliation and rejected-bar gap boundaries. Future-return magnitude is
  **never** a rejection condition. Warm-up rows remain in history.
- Price adjustment UNKNOWN and current-listing survivorship bias remain real
  limitations. Historical fetched-at is retained, not backdated to simulate PIT.
- Gap calendar: observed source bars, conservative weekday-gap segmentation;
  this is not an exchange-holiday certification. Long holidays may conservatively
  reset a window. Labels refer to source observations, not guaranteed tradable fills.
- Expanding chronological folds with session gaps, label purging, separate
  model-selection and calibration blocks. Scaling/clipping fit train endpoints only.
- Small LSTM, validation early stop, validation-only seed selection on latest fold,
  disjoint calibration shrinkage. Batched windows avoid allocating all sequences.
- Zero, persistence, train mean/median, NumPy Ridge and optional sklearn tree use
  the **same** test endpoints. An absent tree is reported and blocks promotion.
- MAE/RMSE, nonflat/balanced direction, flat proportion, activity subgroups,
  date-block paired CI and observed interval coverage are reported. Calibration
  residual ranges are empirical, not guaranteed future confidence intervals.

## Local build, without downloads

Run from the repository root on drive D:

```powershell
.\.venv\Scripts\python.exe scripts\build_lstm_dataset_v3.py --output .runtime\ml\v3\dataset-1
.\.venv\Scripts\python.exe scripts\evaluate_ml_baselines.py --data .runtime\ml\v3\dataset-1\vci_ohlcv_lstm_v3.csv --output .runtime\ml\v3\baselines-1.json
```

First command writes **new** `vci_ohlcv_lstm_v3.csv`, `quarantine.csv`,
`dataset.manifest.json`, a runtime ZIP and `protocol.json`. Existing output
directories are not silently overwritten. Upload these files and the new
`notebooks/train_lstm_kaggle_v3.py` privately to Kaggle only if source rights allow.
Import the new `.ipynb` wrapper or paste the **.py** code, never raw notebook JSON.
Enable a GPU and run the 3-epoch pilot before full training. No Torch/TensorFlow
installation into the application is needed merely to prepare the data.

The training runtime ZIP contains the exact shared causal code, not app secrets.
Its filename/hash are read from `dataset.manifest.json` and verified before
import. For the prepared dataset use **`ml_v3_runtime.aed7aa6849a0.zip`**;
the old `ml_v3_runtime.zip` was retained for preparation audit, not for this run.
If engineering code changes before training, explicitly run
`scripts/build_lstm_dataset_v3.py --output .runtime/ml/v3/dataset-1 --refresh-runtime`.
This packages a content-addressed new ZIP, updates the manifest with history,
and preserves both the CSV and previous ZIP; it never upgrades validation.
Full artifact export contains pinned
`model.keras`, `spec.json`, `evaluation.json`, `protocol.json`, `manifest.json`.
Save the entire directory under D, separate from V2. A rejected/pilot artifact
can remain a labeled shadow estimate, with **zero** V3 Technical weight.

## Promotion and rollback

Protocol: `evaluation/configs/lstm-v3-protocol.json`. Minimum three outer folds,
two winning folds, no excess loss vs Ridge/tree, verified engineering checks,
adjusted compatible input and a **new independent 40-session forward holdout**
with positive paired MAE and directional improvement confidence bounds.
Old repeatedly viewed tests are development evidence, not independent holdouts.
Multiple seeds do not count as multiple independent folds.
Full runs must include every pre-registered seed/fold; non-finite/mistyped
metrics, absent tree results and zero calibrated LSTM signal cannot grant blend.

`scripts/evaluate_lstm_v3.py` checks a forward ledger's artifact identity and
capture/label ordering, and writes a **new review report**. These local timestamps
are not an external attestation. Engineering/human review remains required;
it never silently rewrites the frozen artifact or grants production quality.
The forward CSV's legacy-named `artifact_identity` column must contain the stable
**`predictor_identity`** from the manifest: SHA-256 of weights/spec/protocol
identities. The release `artifact_identity` additionally includes evaluation and
therefore changes when a reviewed forward report is packaged. Keeping the
predictor ID separate prevents an evaluation-release identity cycle. The loader
rejects forward evidence tied to any other predictor.

Local baseline output `.runtime/ml/v3/baselines-2.json` covers all six
fold/horizon combinations, including an actually evaluated CPU tree. These
already-viewed historical observations are development-only. The baseline
results do not imply the untrained V3 LSTM beats zero, Ridge or tree. Optional
tree import/resource failures are reported explicitly and block promotion,
while preserving the other baseline results.

V3 selection is opt-in; default V2 remains available for rollback. V3 policy
eligibility false always means weight 0, even though files load as TRAINED_MODEL.
One/five-session forecasts must never be relabeled three-month target prices.

## Evidence still required

Review of the supplied Kaggle run, independent forward outcomes, price-adjusted
source verification, engineering report approval and live latency remain pending.
Passing fixture tests demonstrates implementation behavior, not predictive skill.
