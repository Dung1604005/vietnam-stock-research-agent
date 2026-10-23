# Nâng cấp Phase 0, 1, 2 — triển khai và kiểm chứng

> Phạm vi báo cáo: số test, latency, corpus và smoke test ghi trong tài liệu là kết quả của lượt triển khai lịch sử, không phải tổng mới ngày 23/10/2026. Trạng thái/code và kiểm chứng mới được tổng hợp tại [project-report.md](project-report.md).

Tài liệu này ứng với **ba phase đầu tiên theo thứ tự trong comprehensive-free-upgrade-plan.md: 0 → 1 → 2**. Không phải Phase 1 → 2 → 3; Knowledge Graph V2 thuộc lượt sau.

## Trạng thái

Phần code nền tảng, NLU/planning và RAG V2 đã được triển khai. Regression: **339 tests passed**. Các gate offline có benchmark tái lập. Chưa đánh dấu đạt các gate cần người dùng thật, corpus doanh nghiệp độc lập hoặc LLM đang chạy thật.

Không cài thêm gói vào C:, không sửa `.env`, không huấn luyện lại LSTM và không tạo dữ liệu thị trường giả. E5 là dependency tùy chọn; môi trường hiện tại chưa cài sentence-transformers/torch hoặc tải trọng số E5.

## Phase 0 — Contracts và evaluation

- `models/research_contracts.py`: RequestBudget, ToolCallRequest, Evidence, AgentFinding, Conflict, FinalAnswer. Timestamp phải có timezone; quantitative finding phải có evidence IDs; final answer không được dẫn ID không tồn tại.
- `chat/router.py`: tái sử dụng ParsedQuery/ RoutingPlan qua tên hợp đồng UserIntent/TaskPlan. Plan có schema version, dependency, input/output, timeout, cost và completion condition.
- `models/state.py`: thêm request_budget tương thích dữ liệu cũ qua default.
- `services/request_telemetry.py`: ContextVar theo request, bộ đếm có lock, giới hạn LLM/tool/deadline trước lần gọi tiếp theo, SQLite telemetry trên D:.
- Đã nối telemetry vào chat, semantic parse, lifecycle agent, tool lấy giá, retrieval, web fetch/search, LLM providers và resilient provider attempts/retries.
- Không lưu query, prompt, payload hay API key vào request telemetry. Token usage chưa được provider adapter thống nhất cung cấp nên là `null`, không được ghi ước lượng thành số thực đo.
- Deadline kiểm tra giữa các lần gọi; không phải cơ chế hủy cưỡng bức HTTP call đang chạy. Ngân sách output token trong contract là cấu hình, chưa phải metering usage token đã xác nhận từ mọi provider.

Benchmark version `upgrade-synthetic-v1` gồm:

| Nhóm | Số lượng | Phạm vi |
|---|---:|---|
| NLU | 320 | 16 mẫu câu × 20 ticker; không phải 320 dạng diễn đạt độc lập |
| RAG | 100 | 80 câu có tài liệu, 20 câu ngoài chủ đề/không có tài liệu doanh nghiệp |
| Corpus RAG kiểm thử | 20 | Nội dung lý thuyết tự soạn, URL example.invalid |
| Screening/cardinality | 50 | Số lượng và horizon |
| Single-stock | 30 | Routing; ngày là metadata tình huống, không phải kiểm chứng giá lịch sử |
| Conflict | 30 | 5 dạng tình huống lặp qua các ticker; kiểm tra action được giao lại |

Các fixture nằm trong `evaluation/upgrade/benchmark-v1.json`, tách khỏi intent example bank và corpus production. Sau khi chạy và phân tích lỗi, bộ này là **synthetic regression/calibration**, không được quảng bá là holdout người dùng độc lập.

## Phase 1 — NLU và Manager

- Cache 256 semantic parse entries, TTL 5 phút, khóa gồm query và ngữ cảnh thực chất. Cache trả bản sao, không chia sẻ list mutable giữa các request.
- Parser cục bộ chạy trước. Configured semantic LLM chỉ được gọi khi confidence thấp, multi-intent hoặc unknown. Provider được inject trực tiếp trong test vẫn được gọi để kiểm tra schema/fallback.
- Manager không gọi LLM routing lại cho intent rõ ràng, confidence ≥ 0.85 và không có secondary intent/clarification.
- Bổ sung DOCUMENT_QA, FORECAST, SOURCE_REQUEST, THESIS_REVIEW và PORTFOLIO_RISK.
- Forecast dùng Data + Forecast tool; không cần Technical/Manager LLM để chỉ trả dự báo.
- Document/source/thesis research dùng RAG và web fallback có giới hạn. Đây là route nghiên cứu evidence; chưa phải tính năng thesis tracker đầy đủ ở Phase 5.
- Portfolio risk được nhận diện và yêu cầu mã/tỷ trọng; phân tích danh mục chat đầy đủ chưa được triển khai ở phase này.
- Nhận diện thêm alias doanh nghiệp, số lượng viết bằng chữ, exchange, as-of/range ngày, risk preference và comparison dimensions.
- Các ngưỡng sàng lọc tùy chỉnh được nhận diện nhưng chưa được bộ screen hiện tại thực thi sẽ dẫn đến clarification, không im lặng bỏ điều kiện.
- Context resolver hiểu “đổi thành 6 tháng”, so với một mã khác và các tham chiếu thứ tự đã có.
- Query nhiều bước có research + technical tạo dependency hợp lệ và thực thi secondary outputs.
- As-of đối với tài liệu được áp dụng vào retrieval. Historical stock analysis trong chat chưa có point-in-time data nên trả giới hạn rõ ràng, không dùng giá hiện tại thay thế.

## Phase 2 — RAG V2

### Ingestion

- Source registry có domain, publisher, quality tier T1–T4, refresh schedule và usage note.
- Domain chính thức là gợi ý chất lượng, không phải cấp phép bản quyền. Manifest ingest cần `approved_full_text` hoặc policy cho phép rõ ràng.
- Network auto-ingest chỉ nhận domain trong registry; kiểm tra cả redirect. Local file có thể ingest khi đã được người dùng duyệt.
- Canonical URL bỏ fragment/tracking query; checksum chuẩn hóa whitespace; ingest lặp giữ nguyên identity và timestamp đầu tiên.
- Exact/near-duplicate có điều kiện bảo toàn numeric tokens. Nội dung sửa tại cùng URL tạo version/supersedes, giữ lịch sử. Revert về nội dung cũ cũng tạo revision mới.
- Migration additive cho SQLite cũ; không xóa bảng/dữ liệu cũ.
- Đọc TXT/MD/HTML và PDF có text layer bằng pypdf. HTML giữ paragraph/table boundaries. Chunker ưu tiên paragraph/section, sau đó word boundary và overlap.
- PDF scan không có text layer trả lỗi yêu cầu OCR; OCR tự động chưa nằm trong phần code đã hoàn tất.
- Đóng kết nối SQLite rõ ràng để tránh giữ file handle trên Windows.

### Retrieval và synthesis

- FTS5/BM25 + vector, Reciprocal Rank Fusion, quality/freshness boost và diversity penalty.
- Ticker, document type, publication date/range và revision visibility được lọc **trước candidate LIMIT**. Điều này sửa trường hợp tài liệu mã cần tìm bị các mã khác chiếm hết top candidates.
- Hash embedding V2 sửa alias khớp một phần từ. Content-anchor gate giảm false positive do bỏ dấu tiếng Việt như mưa/mua, hoa/hóa. Hash embedding vẫn là baseline, không được gọi là pretrained semantic model.
- E5 adapter hỗ trợ query:/passage: prefix, CPU, lazy load, cache trên D:, download opt-in và model_id riêng. Chưa đo chất lượng E5 bằng model thật trong lượt này.
- Context pack giới hạn 6.000 byte UTF-8; source metadata và source IDs đi cùng đoạn trích.
- Structured LLM claims phải có citation ID hợp lệ, tiếng Việt và numeric grounding. Không có evidence thì không gọi LLM.
- FinalAnswer chứa findings và Evidence có provenance.
- Web fallback có thể synthesize đoạn trích đã fetch khi nguồn có published timestamp; nguồn thiếu ngày vẫn chỉ hiện evidence/discovery, không bịa ngày công bố và không tự lưu vào verified store.
- Kiểm tra numeric/content anchors là kiểm tra cơ học. Nó không chứng minh mọi paraphrase/quan hệ nhân quả/ngữ nghĩa được citation hỗ trợ. Citation precision thực tế cần đánh giá riêng.

## Cách chạy

Benchmark toàn bộ, không cần market API hoặc LLM:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_upgrade.py --strict
```

Report mặc định: `.runtime/evaluations/upgrade-latest.json`.

Regression tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Ingest báo cáo đã được phép sử dụng:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_research_manifest.py config\research_manifest.example.json
```

File example có danh sách rỗng để không tự ingest nguồn chưa xác minh. Tạo manifest riêng với cấu trúc:

```json
{
  "documents": [
    {
      "source_url": "https://your-approved-source.example/report.pdf",
      "title": "Tên báo cáo thật",
      "publisher": "Đơn vị phát hành thật",
      "published_at": "2025-12-31T09:00:00+07:00",
      "document_type": "annual_report",
      "tickers": ["FPT"],
      "approved_full_text": true,
      "usage_note": "Cơ sở cho phép sử dụng tài liệu",
      "local_path": "report.pdf",
      "refresh_hours": 72
    }
  ]
}
```

Đây chỉ là **ví dụ schema**, không phải nguồn/báo cáo thực tế. `local_path` tương đối tính từ thư mục manifest. Bỏ local_path để fetch HTTPS từ registry. `--force` bỏ qua TTL fetch; checksum vẫn ngăn duplicate.

E5 tùy chọn; chỉ cài khi muốn dùng pretrained embeddings và có đủ dung lượng trên D:

```powershell
New-Item -ItemType Directory -Force .tmp, .cache\pip
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip --cache-dir .cache\pip install -e ".[rag]"
$env:STOCK_AGENT_RAG_EMBEDDING_PROVIDER = "e5"
$env:STOCK_AGENT_RAG_ALLOW_MODEL_DOWNLOAD = "true"
.\.venv\Scripts\python.exe scripts\reindex_research_vectors.py
```

Model sẽ được cache dưới `.runtime/models/embeddings`. Sau khi tải, có thể đặt `STOCK_AGENT_RAG_ALLOW_MODEL_DOWNLOAD=false`. E5 không hoạt động thì FTS fallback vẫn dùng được; không trộn vector khác model_id trong cùng phép similarity.

## Kết quả trước/sau trên cùng synthetic regression

Baseline lấy từ snapshot Git `b3cecaf`. Runtime reports lưu ở `.runtime/evaluations/upgrade-before.json` và `upgrade-after.json`.

| Metric | Baseline | Sau nâng cấp |
|---|---:|---:|
| Intent accuracy | 75% | 100% |
| Ticker extraction | 100% | 100% |
| Requested count/horizon | 100% | 100% |
| Recall@5 | 86,25% | 96,25% |
| MRR | 0,7085 | 0,9375 |
| Retrieval abstention (20 negative cases) | 50% | 100% |

Kết quả này chưa đo: citation precision của LLM thực, E5 thực, latency mạng P95, khả năng đọc PDF bảng phức tạp, OCR, hay độ đúng trên báo cáo doanh nghiệp độc lập. Không dùng các số regression này để tuyên bố đạt độ tin cậy production.

## Các gate còn cần dữ liệu/đánh giá ngoài fixture

1. NLU holdout người dùng độc lập với dạng diễn đạt mới.
2. RAG holdout gắn với corpus doanh nghiệp thật và nhãn relevant chunk được rà soát.
3. Citation precision ≥ 90% và unsupported-answer abstention ≥ 95% khi chạy LLM thật, gồm claim sai ngữ nghĩa dù số/URL hợp lệ.
4. 30 single-stock cases thực tại các thời điểm có dữ liệu point-in-time.
5. Đánh giá E5 so với hash/FTS trên cùng corpus trước khi đổi default provider.

Các gate này được giữ ở trạng thái chưa đo; không thay bằng kết quả fake hoặc mặc định đánh dấu pass.
