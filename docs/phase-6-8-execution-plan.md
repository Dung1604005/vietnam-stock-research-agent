# Kế hoạch triển khai Phase 6–8 — LSTM, fine-tuning và demo

Ngày lập: 17/10/2026. Cập nhật 18/10/2026: **CODE_IMPLEMENTED — GPU_TRAINING / HUMAN_REVIEW / MODEL_GATE_PENDING**.

Đã triển khai các nền tảng Phase 6–8, giữ V2; dataset V3 riêng đã được tạo. Notebook fine-tune dùng **`Qwen/Qwen3-4B-Instruct-2507`** theo lựa chọn mới của người dùng. Hướng dẫn và bằng chứng kiểm thử: [phase-6-8-implementation.md](phase-6-8-implementation.md). Phần “hiện trạng làm cơ sở” bên dưới là khảo sát trước triển khai, không mô tả trạng thái mới. Mốc kế hoạch ban đầu chưa có GPU run; về sau đã có review artifact LSTM V3/V4, chưa promote V3; Qwen adapter còn gate riêng. Xem model card/review hiện hành thay vì coi phần khảo sát ban đầu là trạng thái hôm nay.

Kế hoạch được tổng hợp sau khi ba subagent ChatGPT trong phiên làm việc khảo sát độc lập các nhánh ML, LLM fine-tuning và UX/vận hành. Đây là phân công trợ lý phát triển, **không phải thêm ba agent vào ứng dụng chứng khoán**.

## 1. Quyết định chung

- Mục tiêu là demo môn học hữu ích, có bằng chứng và khả năng tái lập; ngân sách dịch vụ bằng 0. Không broker, không đặt lệnh, không hứa lợi nhuận.
- Giữ nguyên dataset, notebook và artifact V2. Model mới tạo tên/thư mục riêng; không thay default hoặc chính sách V2 âm thầm.
- Bắt đầu phần nền Phase 8 và Phase 6 song song. Phase 7 bắt đầu bằng đo baseline/chuẩn bị dữ liệu; **chỉ train khi có lý do đo được**.
- Fine-tuning không thay thế RAG, không dạy model ghi nhớ giá hiện tại, không giao phép tính chỉ báo/tài chính cho LLM.
- Trên máy Windows: dependency, venv, cache, dữ liệu, model và temp ở D. Training cloud dùng filesystem của notebook; khi tải kết quả về máy, lưu vào D. Không cài Torch/PEFT vào app chỉ để train ở cloud.
- Phần đã có ở Phase 0–5 phải được tái sử dụng. Tham chiếu [đợt sửa Phase 3–5](phase-3-5-reliability-remediation.md); không coi 525 test fixture là bằng chứng chất lượng live hoặc đầu tư.

## 2. Hiện trạng làm cơ sở

| Chủ đề | Đã có trong repository | Khoảng trống được xác minh |
| --- | --- | --- |
| Dataset | `.runtime/ml/vci_ohlcv_lstm_1m.csv` và metadata: 1 triệu dòng, 991 mã; cleaner riêng | Adjustment UNKNOWN, universe hiện tại không PIT; cleaner bỏ theo `next_session_return` tương lai nếu dùng nguyên cho test sẽ tạo selection bias |
| LSTM | `notebooks/train_lstm_kaggle_v2.py`: 22 feature, window 30, horizon 1/5; train-only scaling, validation calibration | Chưa có walk-forward; train lọc hàng trước khi tạo window; runtime thay hai feature cross-section bằng train-mean proxies |
| Artifact/inference | `lstm_forecast.py` kiểm hash/schema; `price_forecast.py` fallback; artifact trong `notebooks/trainResult` | Gate cũ là single-holdout; chưa có registry/gate V3, calibrated interval hay train-serving parity chung |
| Technical | `_apply_lstm_blend` đang dùng 20% forecast hợp lệ `TRAINED_MODEL` một phiên | Không được mô tả rằng V2 đã qua gate V3; manifest V2 ghi “display only” không khớp chính sách blend hiện tại |
| LLM | Colab Qwen, gateway, prompt role, schema/claim validator, bounded follow-up | Chưa có training-data curation, adapter serving identity/cache isolation hoặc A/B fine-tune |
| Evaluation | Synthetic regression, forward capture, evaluation registry, frozen real snapshot review | Old test đã xem nhiều lần; telemetry không có training conversations; snapshot chưa được chấm không phải gold label |
| UI/report | Chart, bảng evidence, MD/HTML, trace, cache/budget, setup D | Progress hiện in trước các bước dự kiến; role bị gắn Done chung; thiếu profile Quick/Deep/Offline, audit JSON và health tổng hợp |
| Offline | Có snapshot kết quả và dữ liệu cache | Snapshot phân tích chỉ giữ last bar, không đủ để tính lại RSI/LSTM; cần replay bundle riêng và chặn network rõ ràng |

File chứng cứ chính: `scripts/build_clean_lstm_dataset.py`, `services/lstm_forecast.py`, `agents/technical_agent.py`, `services/request_telemetry.py`, `services/review_evaluation.py`, `services/research_workbench.py`, `app/streamlit_app.py`, `ui/components.py`. Các đường dẫn rút gọn thuộc `src/stock_multi_agent/` nếu không ghi rõ.

Tài liệu cũ có nhắc V4/V4.1 nhưng các script/notebook đó không có trong checkout đã kiểm tra. Không lập kế hoạch dựa trên giả định có thể chạy lại chúng ngay.

## 3. Phân công và điểm tích hợp

| Nhánh triển khai sau khi duyệt | Phụ trách | Đầu ra | Không tự thay đổi |
| --- | --- | --- | --- |
| Agent ChatGPT ML | Phase 6: data/features/folds/baselines/LSTM/registry | Experiment tái lập, model card, gate và adapter V3 riêng | Artifact/default V2, policy Manager, model LLM |
| Agent ChatGPT LLM | Phase 7: curation/baseline/QLoRA/A-B | Dataset đã review, báo cáo A/B, adapter nếu đủ điều kiện | Numeric/claim validators, quyền tool và score |
| Agent ChatGPT UX | Phase 8: profiles/events/evidence/health/replay/export | Demo thật, có giới hạn nguồn và hoạt động offline | Tuyên bố model pass, đổi dữ liệu gốc |
| Agent chính | Chốt contract, review chéo, tích hợp, regression, tài liệu | Một bản app tương thích và checklist nghiệm thu | Không tự nâng mọi trạng thái thành hoàn thành |

Trước khi code cần chốt các contract dùng chung: model/adapter identity, validation status, signal weight, execution profile, data origin và audit schema. Một owner sửa mỗi file dùng chung; các nhánh còn lại đề xuất thay đổi để tránh ghi đè.

## 4. Phase 6 — LSTM/ML V3 có kiểm định

### 4.1. Mục tiêu và phạm vi

Tạo phiên bản dự báo có dữ liệu/feature nhất quán giữa train và inference, đo công bằng với baseline và chỉ tác động Technical khi qua gate đã chốt. Không đặt mục tiêu “bắt buộc train cho pass”.

V2 hiện có mức giảm MAE tương đối rất nhỏ so zero-return: khoảng 0,0842% một phiên và 0,4438% năm phiên. Đây không phải tỷ suất lợi nhuận. Giữ nó làm mốc demo/rollback, không xem test cũ là holdout độc lập mới.

### 4.2. Work packages

| ID | Công việc cụ thể | File dự kiến tạo/sửa | Tiêu chí xong |
| --- | --- | --- | --- |
| P6.0 | Đóng băng V2/checksum và protocol: target, thời điểm dự báo, horizons, splits, primary metric, calibration, seeds và gate | Mới `evaluation/configs/lstm-v3-protocol.json`, `docs/lstm-v3-model-card.md` | Có config được chốt trước khi nhìn test; tài liệu ghi đúng V2 đang blend20%; không sửa manifest/hash cũ để che mâu thuẫn |
| P6.1 | Builder V3 có nguồn/unit/calendar/known-at; dedupe, gap, quarantine, corporate-action status và count reconciliation | Mới `scripts/build_lstm_dataset_v3.py`, `services/ml_dataset_v3.py`; output `.runtime/ml/v3/` | Không overwrite raw/clean V1; không loại sample test chỉ vì label return lớn; UNKNOWN vẫn UNKNOWN |
| P6.2 | Một feature engine dùng chung train/runtime; window liên tục, warmup và endpoint eligibility riêng | Mới `services/ml_features_v3.py`, tests parity/causality | Golden train/inference feature giống nhau theo tolerance; thay dữ liệu tương lai không làm đổi feature quá khứ |
| P6.3 | Expanding walk-forward theo lịch phiên, purge nhãn vượt biên và gap theo max horizon | Mới `services/ml_validation.py`, split manifest | Mỗi scaler/clipping/vocabulary chỉ fit train; không có train label nhìn sang validation/test |
| P6.4 | Zero, return persistence, train-only mean/median, Ridge, tree nhẹ trên cùng endpoint/horizon | Mới `services/ml_baselines.py`, `scripts/evaluate_ml_baselines.py` | Báo MAE/RMSE raw return, cùng mẫu; không chọn baseline từ test |
| P6.5 | LSTM nhỏ, two-horizon, early stopping, validation-only calibration; generator theo batch | Mới `notebooks/train_lstm_kaggle_v3.py`, notebook wrapper và config | Smoke trước full; checkpoint/resume; export cả model reject nhưng nhãn đúng; không có all-window RAM explosion |
| P6.6 | Fold/horizon/subgroup reports, paired date-block bootstrap, interval calibration, registry và model card | Mới `scripts/evaluate_lstm_v3.py`, `services/ml_model_registry.py` | Có report/checksum/config/code revision và limitation; test cũ chỉ development; independent forward holdout lưu riêng |
| P6.7 | Shadow V3, contract serving, eligibility theo horizon/data; toggle lựa chọn và rollback | Mới `services/lstm_forecast_v3.py`; sửa `models/forecast.py`, `price_forecast.py`, `technical_agent.py` và tests | Default V2 giữ nguyên; V3 không đủ điều kiện → 0% blend; khi đạt → tối đa20%; LLM không được sửa forecast |

### 4.3. Thiết kế dữ liệu và feature

1. Mỗi sample lưu `forecast_origin`, `feature_end`, `target_end`, nguồn và thời điểm dữ liệu thực sự khả dụng. Dự báo từ dữ liệu đóng cửa t không được gọi là dự báo trước đóng cửa t.
2. Chuẩn hóa unit/price-adjustment theo khai báo, không suy adjusted từ biểu đồ mượt. Có thể làm experiment lớp học trên UNKNOWN; không biến nó thành strict investment validation.
3. Không coi mọi biến động >20% là lỗi chắc chắn. Quarantine lý do và nguồn; không xóa label khó khỏi test để giảm MAE.
4. Tạo window từ chuỗi đủ ngữ cảnh trước, sau đó mới lọc endpoint bằng điều kiện biết tại t. Không nối qua gap chưa xác minh hoặc ghép hai đoạn bị loại thành 30 phiên giả liên tục.
5. V3a ưu tiên feature single-ticker thực sự tính được ở runtime. Bỏ hai cross-section feature nếu không có snapshot đồng bộ; không train bằng giá trị thật rồi runtime lặng lẽ điền mean. Parity phải kiểm cả feature dẫn xuất như `relative_return_1`, không chỉ hai cột proxy gốc.
6. V3b mới thêm benchmark/sector khi có cùng as-of và đủ coverage. Thiếu context phải từ chối/đổi model tương thích, không tự suy dữ liệu.
7. Chọn universe theo thông tin đã biết ở train origin; universe current-listing vẫn có survivorship bias và không được gọi là PIT. Nếu chưa có nguồn miễn phí hỗ trợ PIT/adjusted thì ghi giới hạn, không fake bổ sung.

### 4.4. Split và evaluation

- Khởi đầu 3 outer folds tiến thời gian. Model/config chỉ được chọn bằng train/validation; outer test không dùng để chọn alpha, bias, architecture hay seed tốt nhất.
- Purge sample có `target_end` vượt biên; gap tối đa horizon5 tính bằng phiên, không `timedelta(5)`. Shared historical lookback trong window test là hợp lệ nếu chỉ chứa quá khứ; không purge máy móc toàn bộ 30 phiên.
- V2/V3/V4 results đã xem chỉ là development evidence. Khóa protocol rồi thu forward observations mới; chờ label đủ 1/5 phiên. Chưa đủ ngày thì `PENDING_FORWARD_VALIDATION`, không tái chế test cũ thành holdout mới.
- Group/block bootstrap theo ngày, giữ dependence giữa ticker và nhãn 5 phiên chồng lấn; không xem hàng trăm nghìn hàng là hàng trăm nghìn quan sát độc lập.
- Direction cần so majority/nonflat baseline thích hợp, báo tỷ lệ phiên 0 và balanced accuracy; không mặc định mọi bài toán đều có baseline50%.
- Range chẩn đoán RMSE không phải calibrated prediction interval. Nếu làm interval, tách calibration split và báo empirical coverage/width theo horizon; không đổi thành phần trăm “tin cậy quyết định”.

### 4.5. Gate và giữ V2

Tách ba trục: `artifact_status` (file/model load được), `validation_status` (bằng chứng kiểm định), `policy_eligibility` (được dùng score hay không). `TRAINED_MODEL` không tự đồng nghĩa đủ chất lượng.

- Engineering gate: schema/hash/unit/as-of đúng, feature parity, causal/no-leak split, deterministic fallback và memory smoke đạt.
- Research regression gate: thắng zero-return MAE trên ít nhất2 outer folds; báo paired improvement/CI; không thua Ridge/tree quá margin đăng ký trước. Chưa có bằng chứng độc lập → chưa được gọi là verified improvement.
- Technical blend gate V3: cần đúng horizon, input đủ chất lượng, regression gate và bằng chứng về chiều dự báo phù hợp với cách chuyển return thành Technical score. Nếu CI hoặc kiểm định không kết luận được thì `INCONCLUSIVE`, không nới gate sau kết quả.
- Demo gate: model chưa đạt vẫn có thể hiện rõ `EXPERIMENTAL`, không thay score/BUY-HOLD-SELL. Phase 6 có thể hoàn tất về engineering dù không có model được promote.
- V2 tiếp tục theo cấu hình hiện hành đến khi có quyết định chuyển riêng. Sidecar model-card/policy giải thích mâu thuẫn manifest display-only và blend20% hiện tại; không sửa lịch sử frozen artifact.
- Không tự diễn giải forecast 1/5 phiên thành target price3 tháng. Report phải hiển thị horizon thực và tác động score thực.

### 4.6. Tài nguyên và giới hạn

CPU laptop làm builder/features/baseline/tests trước; GPU notebook dành training. Pilot 50k–100k endpoint trước full. Một kiến trúc chốt, tối đa3 folds ×2 seeds ban đầu; seed không làm tăng số test độc lập. Nếu cần thêm thử nghiệm phải ghi số lần thử.

Tensor 1M×30×22 float32 riêng nó khoảng2,64GB, chưa gồm copies/masks/pandas. Dùng batch generator/memmap/shards, không tạo toàn bộ rồi copy theo split. Một GPU trước; T4×2 không tự tăng tốc. Chỉ ước lượng thời gian train sau khi đo pilot trên GPU được cấp.

## 5. Phase 7 — Fine-tuning LLM có điều kiện

### 5.1. Quyết định đầu vào

Không train một model cho mỗi agent. Chọn **một tác vụ có lỗi đo được**: ưu tiên NLU/typed routing hoặc structured financial extraction; synthesis là bước sau. Nếu lỗi do corpus thiếu/cũ hoặc schema/prompt sửa được thì giữ base + RAG.

`request_telemetry.py` chủ ý không lưu prompt; frozen snapshots chưa review không phải training labels. Cần pipeline curation mới có opt-in, không bật logging hội thoại riêng mặc định.

### 5.2. Work packages

| ID | Công việc | File dự kiến | Gate |
| --- | --- | --- | --- |
| P7.0 | Baseline raw model và end-to-end; freeze prompt/schema/base revision/corpus; chọn task/metric | Mới `config/llm-training-v1.json`, `docs/llm-finetuning-data-policy.md`, `services/llm_evaluation.py` | Có error analysis thật; thiếu justification → kết thúc `NO_TRAINING_JUSTIFIED` |
| P7.1 | Opt-in export, redact, quyền nguồn, dedupe, approval UI/JSON và curator review | Mới `models/finetuning.py`, `services/finetuning_dataset.py`, `scripts/prepare_llm_finetuning_dataset.py` | Không secret/PII; chỉ approved labels có quyền dùng; không đọc `.env` làm dataset |
| P7.2 | Group splits, stress views, contamination audit và lock digests | `.runtime/llm_training/v1/*.jsonl`, public-safe manifest trong `evaluation/llm-finetuning-v1/` | Không cùng conversation/template/document revision vượt split; test tách tuning |
| P7.3 | QLoRA pilot một task trên notebook miễn phí, checkpoint và version pin sau smoke | Mới `notebooks/finetune_qwen3_4b_lora_v1.py` và `.ipynb` | Pilot memory/throughput ổn; không full fine-tune, không cài stack training vào app |
| P7.4 | A/B base vs adapter trên cùng input, decoding và quantization; human blind review nếu synthesis | Mới `scripts/evaluate_llm_adapter.py`, model card và tests | Đạt gate đăng ký trước, không giảm grounding/abstention/safety |
| P7.5 | Optional adapter loader, model identity, cache isolation, health và one-switch rollback | Notebook serving bản mới; sửa provider/gateway; thêm config example | Hash/base/tokenizer/schema mismatch bị từ chối; fallback base/deterministic; default adapter off |

### 5.3. Dataset

Mục tiêu500–2.000 ví dụ **đã duyệt**, bắt đầu bằng tập nhỏ để kiểm curation. Số lượng không thay thế chất lượng. Mỗi ví dụ có:

`example_id`, `task`, `input/messages`, `approved_output`, `schema_version`, `prompt_version`, `issuer_group`, `template_group`, `document_family`, `available_at`, source IDs/checksum, quyền sử dụng, label origin, reviewer, reviewed_at và approval status.

- Bao phủ không dấu/typo, multi-turn, “1/10 mã”, horizon, nhiều ý, hỏi lại khi thiếu mã, kỳ quý/năm/YTD, triệu/tỷ, nguồn mâu thuẫn, thiếu evidence và prompt injection.
- Model có thể tạo nháp/paraphrase gắn `synthetic_draft`; không tự coi output hiện có là gold. Negative example phải dạy abstain/clarify/reject đúng.
- Dữ liệu report truy cập công khai chưa chắc được quyền train/phân phối. Chỉ nguồn/quyền phù hợp hoặc trích đoạn được phép; kiểm trước upload notebook.
- Redact key/header/token, URL query chứa secret, email/điện thoại/account, danh mục cá nhân, đường dẫn local và private conversations. Manual preview + secret scan trước cloud upload.
- Split khoảng70/15/15 **theo nhóm**, không random từng câu. Tối thiểu hướng tới100 test cases cho task chính nếu đủ dữ liệu; báo CI/sample size.
- NLU ưu tiên unseen template/context/issuer; fact extraction ưu tiên document family/issuer/time. Tạo stress views riêng khi giao cả ba trục làm split rỗng; công bố overlap, không tuyên bố đồng thời độc lập nếu chưa đạt.
- Bộ synthetic regression hiện hữu không được vừa đưa train vừa gọi test độc lập. Test đã xem để sửa prompt/model trở thành development.

### 5.4. Pilot và go/no-go

Candidate được người dùng chốt, **không phải cam kết fit GPU trước smoke**: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; 4-bit NF4 + LoRA rank8, FP16 cho T4, sequence512, microbatch1 và accumulation8, gradient checkpointing và answer-only loss. Chỉ tăng context sau khi đo đủ RAM. Giữ chat template/tokenizer đồng nhất. Notebook pin thư viện và có CUDA readiness; vẫn cần GPU smoke thực. Hướng này dựa trên [tài liệu PEFT chính thức](https://huggingface.co/docs/peft/main/en/developer_guides/quantization).

Chạy50–100 steps pilot, ghi peak memory/examples per second rồi mới ước lượng toàn bộ. Thử1–3 epochs, chọn bằng validation metric; checkpoint/resume/export về D. Không ép dùng cả hai GPU hoặc train mọi agent.

Gate đăng ký trước:

1. Metric task chính tăng≥5 **điểm phần trăm** trên grouped unseen holdout; hoặc nếu baseline gần trần, chọn trước mục tiêu giảm≥20% latency/output token kèm non-inferiority. Không đổi mục tiêu sau kết quả.
2. Raw JSON/schema validity mục tiêu≥98%; output runtime được nhận vẫn phải qua validator100%. Báo cả raw model lẫn end-to-end để fallback không che lỗi.
3. Không giảm issuer/period/unit/source accuracy, abstention, numeric grounding; tolerance phải chốt trước. Không có lỗi critical mới về injection/secret/tool quyền.
4. Paired/grouped CI và blind human review khi cần. CI chưa kết luận hoặc dataset nhỏ → `INCONCLUSIVE`; không dùng model tự chấm thay người.
5. Artifact manifest pin base/tokenizer revision, chat-template/schema/prompt hashes, dataset splits, seed/config/lib versions, metrics/checksum/license. Không public adapter/dataset mặc định.

Kết thúc Phase7 hợp lệ theo hai cách: `COMPLETED_WITH_APPROVED_ADAPTER`, hoặc `COMPLETED_NO_ADAPTER_JUSTIFIED` với report minh bạch và giữ base+RAG. Loss giảm không đủ để tuyên bố assistant tốt hơn.

## 6. Phase 8 — UX, vận hành và đóng gói demo

Không chờ model mới train xong mới làm Phase8. Đây là nhánh mang giá trị demo sớm nhất.

### 6.1. Work packages

| ID | Nội dung | File dự kiến | Nghiệm thu |
| --- | --- | --- | --- |
| P8.1 | Request-scoped profile Quick/Deep/Offline và origin theo nhóm dữ liệu | Mới `models/execution_profile.py`, `services/execution_profiles.py`; sửa chat/models/UI | Không sửa global `.env` khi đổi mode; mixed live/cache/snapshot/model không bị một nhãn chung che giấu |
| P8.2 | Event STARTED/SUCCEEDED/PARTIAL/FAILED/SKIPPED/FALLBACK thật theo node/attempt | Event model/sink, telemetry, graph, app/components | Không in trước kế hoạch như đã chạy; lỗi không thành Done; rerun không nhân request |
| P8.3 | Unified claim→quote→source drawer; sanitizer và versioned audit JSON; dùng chung cho MD/HTML | Mới `services/audit_export.py`; sửa exporter/components | Không full dump metadata/request_parameters; kiểm sentinel secrets và HTML injection; user xem preview trước export |
| P8.4 | Health/readiness local-first; explicit cheap probes có TTL | Mới `services/runtime_health.py`, UI panel | Render/rerun không tự gọi giá/LLM; UNKNOWN khác HEALTHY; Neo4j/model tùy chọn không làm sập app |
| P8.5 | Replay bundle riêng có OHLCV/facts/provenance/rights/checksum và default-deny offline provider | Mới `models/demo_bundle.py`, `services/demo_bundle.py`, `providers/offline_provider.py`, `scripts/capture_demo_bundle.py` | Offline không gọi mạng; không thay fixture vào; giữ original timestamps, ticker không có trả thiếu |
| P8.6 | Setup/D-drive, CI smoke, guide5–10phút và measured latency table | Mở rộng setup; mới `docs/demo-runbook.md`, tests/AppTest | Fresh setup không tải ML/E5 mặc định; missing optional services không crash; rehearsal chạy được thật |

### 6.2. Profiles và provenance

Đây là **budget thiết kế**, không phải latency đã đo:

- Quick: deterministic/cache/KB trước, mặc định0 LLM; người dùng chọn rõ thì tối đa1 call. Mục tiêu20–30giây với cache phù hợp. Screening có ngân sách riêng, không tự deep100 mã.
- Deep: tối đa4 calls/180giây như profile hiện có; mở rộng6 calls khi người dùng chọn. Web/RAG/KG và follow-up đều chia sẻ trần, không mỗi agent một quota riêng.
- Offline:0 outbound/LLM calls; chỉ bundle đã xác minh/KB local. Không tự đổi snapshot cũ thành giá hiện tại.
- Giữ hành vi hiện hành cho tới khi selector/profile mới được nghiệm thu; không âm thầm đổi mọi phân tích sang Quick.

Origin thể hiện riêng: `LIVE_FETCH`, `CACHE`, `SNAPSHOT_REPLAY`, `MODEL_ESTIMATE`, kèm original as-of/fetched/publication/reporting-period/capture/generation time. Cache access không được cập nhật lại thời điểm nguồn. Replay text LLM cũ không phát event như đang gọi model mới.

### 6.3. Progress, health và privacy

- Progress lấy từ event sink hoặc graph stream; worker chỉ phát event qua queue/thread-safe context, không gọi Streamlit UI trực tiếp. ID gồm request/node/attempt/sequence để phân biệt parallel/rerun. Không dùng phần trăm tiến độ giả khi task plan đổi.
- Khi deadline/failure, đóng trạng thái task và giữ phần hoàn thành có provenance; cancellation là cooperative trước tool tiếp theo, không hứa ngắt tùy ý một HTTP call đang block.
- Health mặc định đọc trạng thái quan sát gần nhất, artifact hash/schema, index/version/count, sync/circuit/cache/disk/D-path. Không load GPU hoặc generate text chỉ để render dashboard.
- Nút probe riêng timeout ngắn, debounce/TTL; functional LLM generation probe là thao tác khác được tính quota. 20UI rerun phải gây0 external health calls.
- Audit allowlist gồm versions/profile/tasks/budget, nguồn/checksum, canonical facts, điểm/contributions, claim audits, follow-up, decision và limitations. Không xuất credentials/headers/raw prompts/hidden reasoning/local username/path/raw exception/request_parameters.
- Sanitize URL query/fragment và chuỗi lỗi; danh mục cá nhân chỉ export khi opt-in. MD/HTML dùng cùng sanitizer. Escape source-derived strings trước `unsafe_allow_html`; chart HTML nhúng chưa có thì xuất specs/bảng, không quảng cáo như đã hỗ trợ.

### 6.4. Offline bundle và bài demo

Snapshot phân tích hiện chỉ có last bar nên **không đủ recompute RSI/LSTM**. Bundle mới phải chứa chuỗi OHLCV cần thiết từ nguồn/cache được phép, facts, trích đoạn hợp lệ hoặc URL metadata, model output đã ghi, manifests/checksums và giới hạn quyền dùng.

Bundle đầy đủ giữ `.runtime/demo/` trên D; Git chỉ giữ manifest/schema/test fixture được ghi rõ và quyền phù hợp. Không public raw report, dataset hay transcript riêng chỉ vì muốn demo dễ hơn. Sai checksum phải bị từ chối; không có ticker/as-of phải báo thiếu.

Runbook5–10phút:

1. Readiness, mode và ngày của dữ liệu.
2. Hỏi lý thuyết có KB citation.
3. Phân tích một mã, quan sát node thực và Technical/Fundamental/Risk.
4. Multi-chart: doanh thu/dòng tiền/RSI và kỳ/đơn vị.
5. So sánh hoặc shortlist nhỏ, giải thích partial count.
6. Xem conflict/follow-up có thật; nếu replay tình huống cũ phải gắn nhãn, không giả live conflict.
7. Tắt optional services, chạy offline, tải report/audit.

## 7. Điều kiện công cụ miễn phí cần chốt trước triển khai

- **Model mới Qwen3-4B-Instruct-2507 công bố Apache-2.0:** kiểm license/revision và quyền dữ liệu trước train/phân phối. Lưu ý lịch sử: Qwen2.5-3B cũ dùng `qwen-research`, không suy license giữa hai model. Nguồn cho model đã chọn: [model card chính thức](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- **Colab free không có cam kết GPU/quota/session.** FAQ còn hạn chế dùng chủ yếu qua web UI ngoài notebook và các web service không thuộc interactive compute. Không mặc định tunnel phục vụ app được phép chỉ vì demo; kiểm quy định trước sử dụng live route. Training và batch A/B tương tác trong notebook là đường chính; app có truthful replay/deterministic fallback. Không dùng keepalive, nhiều tài khoản hoặc cách né hạn mức. Nguồn: [Colab FAQ](https://research.google.com/colaboratory/faq.html).
- Không yêu cầu thuê GPU/API. Nếu phiên miễn phí không đủ tài nguyên thì checkpoint/resume, giảm cấu hình/thí nghiệm hoặc ghi chưa đủ; không tự chuyển sang dịch vụ tính tiền.
- Không cài model mới vào C. Dataset/model lớn giữ local D/private cloud notebook; quyền upload phải được kiểm, không tự chia sẻ công khai.

## 8. Thứ tự, milestone và dependency

```text
M0: Freeze contracts / V2 / protocols / rights / baseline
 ├─ P8.1–3: Profiles + real events + sanitized evidence/report
 ├─ P6.1–4: New data + shared features + splits + baselines
 └─ P7.0–2: LLM baseline + opt-in curation + locked grouped splits
          ↓
M1: Offline-safe demo + reproducible ML/LLM evaluation packages
 ├─ P6.5–6: Kaggle LSTM + walk-forward + forward capture
 ├─ P8.4–6: Health + replay bundle + rehearsal
 └─ P7.3–4: QLoRA only if baseline/dataset justify it
          ↓
M2: Model/adapter gates + optional shadow / integration / rollback
          ↓
M3: Offline acceptance + measured live run + human-reviewed report
```

Ước lượng để lập lịch, không phải cam kết: Phase6 khoảng3–5tuần kỹ thuật/validation; Phase8 khoảng6–10ngày tập trung có thể song song; Phase7 nếu justified khoảng2–4tuần, phần lớn công là review dữ liệu. Forward holdout có thời gian chờ nhãn/thị trường riêng, không rút ngắn bằng GPU nhanh hơn. Không hứa thời lượng GPU trước pilot.

Ưu tiên lần code đầu: P6.0–3, P7.0–1 và P8.1–3. Chưa bắt đầu bằng tải model lớn, grid search hoặc thay V2. Sau mỗi milestone chạy regression, review diff và cập nhật phần đo thật/chưa đo.

## 9. Test matrix và Definition of Done

- Phase6: duplicate/gap/unit/known-at/adjustment; no future feature influence; train/runtime parity; no label overlap; scaler train-only; baseline same rows; artifact corruption; incompatible feature order/vocab/scaler; missing benchmark; eligibility0/20% và fallback/rollback.
- Phase7: redaction và quyền dữ liệu; dedupe/group leakage; schema/claim/issuer/unit/period negative cases; adapter hash/revision/cache isolation; injection/abstention; raw-vs-fallback metrics; không mất validator sau fine-tune.
- Phase8: Quick/Deep/Offline AppTest; patched outbound transports bị chặn; LLM/Neo4j/Web tắt; cache expired; source stale; artifact hỏng; index sai embedding version; DB readonly; partial provider; real event order và duplicate rerun; exports không rò secret/HTML injection.
- Phân loại riêng `CODE_COMPLETE`, `OFFLINE_ACCEPTED`, `LIVE_PENDING/PASS`, `HUMAN_REVIEW_PENDING/PASS`, `MODEL_GATE_PENDING/PASS/REJECTED/INCONCLUSIVE`. Không gộp thành một checkbox “model tốt”.
- Chỉ chuyển default V3 hoặc adapter khi đúng gate, có model card/report/checksum, resource smoke và rollback pass. Không pass thì giữ V2/base+RAG vẫn là kết quả nghiên cứu hợp lệ.

## 10. Trạng thái triển khai cập nhật 18/10/2026

- [x] Tạo3 subagent ChatGPT và khảo sát code theo3 nhánh.
- [x] Tổng hợp hiện trạng, risk, work packages, dependencies, file dự kiến và nghiệm thu.
- [x] Kiểm nguồn chính thức về PEFT, model license và giới hạn Colab.
- [x] Cả ba subagent đã review lại phần kế hoạch tương ứng, không ghi nhận factual blocker.
- [x] Triển khai code nền Phase6–8; kiểm thử offline và UI, tài liệu riêng bên dưới.
- [x] Tạo dataset V3 riêng từ nguồn 1 triệu dòng; giữ raw/V2.
- [x] Tạo notebook Qwen3-4B load/train/export/reload và pipeline dataset/A-B/registry.
- [ ] Fine-tuning dataset đúng task/schema app được con người duyệt (pilot tự sinh chỉ là demo).
- [ ] Train notebook / đo A-B / independent holdout / promote artifact.

Đợt triển khai không sửa `.env`, không cài training dependencies trên laptop, không tải model hoặc train cloud thay người dùng, không commit/push. V2 và base+RAG là rollback. Xem [hướng dẫn triển khai](phase-6-8-implementation.md), [fine-tuning](llm-finetuning-data-policy.md) và [demo runbook](demo-runbook.md).
