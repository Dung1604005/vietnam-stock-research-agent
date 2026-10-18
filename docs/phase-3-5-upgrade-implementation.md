# Triển khai Phase 3–5

Ngày bắt đầu: 17/10/2026. Không thêm dịch vụ trả phí, không thay artifact LSTM V2,
không kết nối broker. Mọi đường dẫn runtime/model/cache giữ trên ổ D.

**Bản bổ sung sau review hạn chế:** xem [kế hoạch và triển khai khắc phục](phase-3-5-reliability-remediation.md).
Bản mới thêm canonical financial facts, claim anchors, bounded LLM evidence follow-up,
Risk PARTIAL/coverage, lịch sử chart và công cụ review snapshot thật. Các số test/backfill
ở cuối tài liệu này là kết quả đợt trước, không phải nghiệm thu live hiện tại.

## Checklist và thứ tự

1. Phase 3: ontology có version; SQLite graph có provenance/thời gian; entity
   resolution bảo thủ; validator cho triple; đồng bộ Neo4j bằng MERGE; Graph-RAG
   bật riêng, không tự thay retrieval mặc định khi chưa đo relational holdout.
2. Phase 4: gateway chung cho provider đã cấu hình (bao gồm Colab Qwen), cache,
   circuit breaker, deadline/call/output budget; prompt contracts; Evidence Pack;
   validation tiếng Việt, nguồn và số liệu; chuyên gia nhận cùng advisor nhưng
   vẫn bật theo role và tổng ngân sách, không gọi tất cả trong mọi request.
3. Phase 5: brief/snapshot và giải thích thay đổi, event impact có điều kiện,
   thay đổi báo cáo giữa kỳ, peer percentile, scenario, thesis tracker,
   portfolio do người dùng nhập, export Markdown/HTML; comparison 2–5 mã có
   alignment gate; screening có filter, shortlist và partial-result reasons.
4. Kiểm thử offline bằng fixture được ghi rõ là synthetic; regression toàn bộ
   suite. Không gán kết quả test fixture cho hiệu quả đầu tư hay độ chính xác
   dữ liệu ngoài đời.

## Acceptance gates cần đo riêng

- Graph: 20 câu hỏi relational có nhãn độc lập, ≥90% answer/citation và cải thiện
  Recall@5 ≥5 điểm phần trăm. Chưa đủ phép đo thì Graph-RAG vẫn opt-in.
- Colab: P95 một mã <60 giây khi endpoint/model hoạt động; warm-cache screening
  100 mã P95 <120 giây. Test fake provider không thay thế phép đo này.
- Numeric grounding là kiểm tra cơ học có giới hạn, không phải chứng minh ngữ
  nghĩa hay đảm bảo model không suy luận sai.

## Phần đã triển khai

### Phase 3 — KB/KG V2

- Models/ontology và validator: `models/graph_v2.py`.
- Ledger/traversal SQLite, alias bảo thủ, node/edge provenance, interval và
  Neo4j MERGE projection: `services/graph_v2.py`.
- Research ingest tự projection; Theory KB đồng bộ incremental; stock brief
  projection structured facts/events/model output có source URL.
- RAG/KB có đường đọc graph opt-in; không phụ thuộc Neo4j khi chạy demo.
- Xem [ontology và mapping](investment-graph-v2.md).

### Phase 4 — LLM có kiểm soát

- `LLMGateway` dùng chung cho provider cấu hình, gồm Colab Qwen. Cache RAM,
  circuit breaker, explicit connectivity probe, timeout theo deadline, retry
  transport có giới hạn, schema/language/numeric/source validation.
- Telemetry model/version/hash/status; không ghi prompt/token/API key. Output
  budget dùng UTF-8 byte upper bound bảo thủ, **không phải token usage thực**.
- Manager typed intent/synthesis hiện hữu đi qua gateway. Research có query
  decomposition tùy chọn, claim/citation RAG, review đối chiếu evidence khi
  expanded. Technical chỉ chọn tool; Python tính chỉ báo.
- Fundamental/Risk nhận advisor chung và có role allowlist. Fundamental có
  fact-mapping quote/unit/period-checked, lưu `PROPOSAL_NOT_SCORED`, không tự
  điền số vào score. Profile demo mới chạy review Fundamental/Risk trên evidence
  cuối sau vòng conflict, kể cả khi các specialists đồng thuận.
- Conflict records có dimension data/time/metric/interpretation/policy; thêm
  phát hiện horizon mismatch/metric conflict; vòng rerun dừng khi hết deadline,
  hết iteration hoặc evidence không tiến triển.
- Prompt/quyền hạn: [agent-prompt-contracts.md](agent-prompt-contracts.md).

Profile demo `.env.example` bật `technical,fundamental,risk`. Có thể đổi
`STOCK_AGENT_LLM_SPECIALIST_ROLES=technical,fundamental` hoặc `fundamental,risk`.
Đây là **quyền được dùng**, không phải bảo đảm mọi role được gọi trong cùng
request: analysis demo tối đa 4 calls (thay profile cũ 2), deep research 3, simple 1.
Xem [hướng dẫn demo chart/LLM](demo-analysis-charts-specialists.md) về ngân sách mới.
Không cần sửa `.env` đang chứa key để chạy các công cụ deterministic mới.

### Phase 5 — Công cụ nghiên cứu

- Stock analysis tạo research brief/snapshot có tool + source IDs theo mục.
- “Vì sao kết luận thay đổi?” đối chiếu hai snapshot cùng ticker/horizon có
  timestamp, score/signal/policy changes; không gán tương quan thành nhân quả.
- Comparison 2–5 mã kiểm tra as_of/currency/price unit/financial period; field
  không tương thích bị giữ lại. Percentile chỉ trong peer set người dùng nhập
  có cùng sector/metric/unit/period/currency, không giả là percentile toàn ngành.
- Financial change theo hai kỳ mới nhất của cùng metric/statement/unit/frequency;
  không gọi QoQ/YoY khi chưa xác minh standalone/YTD và cơ sở kế toán.
- Event impact có nguồn/ngày; tác động chỉ là điều kiện nghiên cứu, chưa có
  mô hình định lượng tác động tin tức. Nếu thiếu events/statements thì tìm
  thêm tài liệu, không tự biến excerpt thành fact.
- Scenario price dùng `close * (1 + shock)`, không phải forecast/target price.
- Portfolio tối đa 20 mã: HHI, vị thế lớn nhất, phần cash, covariance/volatility
  trên lịch sử phiên đồng thời. Không normalize tỷ trọng âm/trùng/tổng >100%.
- Thesis tracker lưu luận điểm/catalysts/risks/evidence/status và lịch sử cập
  nhật; khi phân tích lại cùng mã, ghi evidence mới/không còn current. Không tự
  coi luận điểm đúng chỉ vì retrieval có kết quả.
- Markdown/HTML download dưới response; HTML escape, không chạy nội dung nguồn.
- Screening giữ cheap technical pass → successive halving → deep shortlist →
  một synthesis cho toàn danh sách. Cache kết quả technical universe 120s;
  filter PE/PB/ROE/RSI và exchange thực thi trên shortlist có ngân sách, kèm lý
  do loại/chưa xét. Không phải exhaustive financial screen toàn thị trường.
- Theory QA có citation vẫn dùng Theory KB hiện hữu, không fetch giá vì hỏi lý thuyết.

## Cách chạy và thử

Không cài thêm package trong lượt này. Khởi động lại Streamlit để dựng service mới:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Ví dụ chat:

1. `Phân tích FPT` — lưu snapshot. Phân tích lại sau khi data/policy thay đổi,
   rồi hỏi `Vì sao kết luận FPT thay đổi?`.
2. `So sánh FPT CMG CTR ELC FOX` — tối đa 5 mã; không bịa trường thiếu.
3. `Tác động sự kiện FPT`.
4. `So sánh báo cáo FPT hai kỳ`.
5. `Kịch bản FPT giảm 5% tăng 10%`.
6. `Rủi ro danh mục FPT 60% HPG 40%`.
7. `Kiểm tra luận điểm đầu tư FPT tăng trưởng nhờ chuyển đổi số`.
8. `Tìm 10 mã cổ phiếu PE dưới 15` hoặc `Tìm cổ phiếu RSI dưới 40 trên HOSE`.
9. `Xuất báo cáo HTML`, hoặc bấm nút download dưới câu trả lời có data.

Giá không có khai báo đơn vị được ghi UNKNOWN/chưa xác minh thang giá, không
tự gắn VND hay nhân 1.000. Provider cần khai báo unit thật trước khi numeric
comparison được mở. Model LSTM/artifact V2 không được train hay thay đổi ở lượt này.

## Lưu trữ, quyền riêng tư và giới hạn

Graph/snapshot/thesis dùng SQLite dưới `.runtime` trên D; không cần download
model mới. `.runtime` bị gitignore. Snapshot/thesis là dữ liệu nghiên cứu người
dùng, lưu cục bộ; bản app này chưa có auth/tenant isolation, không nên public
nhiều người dùng để lưu danh mục riêng tư.

Không cài Neo4j server hoặc model E5 mới. `--neo4j` chỉ đồng bộ nếu server đã
được người dùng cấu hình. Không tuyên bố online Colab/Brave/market provider
đang hoạt động chỉ từ test fake. Các gate hiệu quả retrieval/P95 đầu tài liệu
vẫn chờ đo thực tế; fine-tuning/LSTM V3 thuộc Phase 6–7, không triển khai ở đây.

## Kết quả kiểm tra ngày 17/10/2026

- Regression: **441 tests passed**, gồm **47 test mới** cho Phase 3–5. Có 20
  query graph synthetic; không coi là independent relational holdout.
- Streamlit AppTest: màn hình chat và thư viện báo cáo đều **0 exceptions**;
  smoke test tắt external model/network, không phải online integration test.
- Backfill kho hiện có: **79 nodes, 361 edges, 83 evidence records**. Chạy lại
  không đổi các số lượng này. Neo4j MERGE được test bằng adapter fake; chưa
  tuyên bố đồng bộ thành công server Neo4j thực.
- `compileall` và `git diff --check` không lỗi. Không cài package mới, không
  sửa `.env` chứa key, không thay artifact LSTM, chưa commit/push.
