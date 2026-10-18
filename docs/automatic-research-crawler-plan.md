# Kế hoạch crawler và auto-ingestion cho Research RAG

Triển khai C0–C4 và hướng dẫn vận hành: [automatic-research-crawler-implementation.md](automatic-research-crawler-implementation.md). Các gate pilot thực tế và C5 tùy chọn được ghi riêng trong tài liệu triển khai.

## 1. Mục tiêu và phạm vi

Mục tiêu là biến pipeline hiện tại từ chế độ “biết URL rồi mới ingest” thành một quy trình tự động:

```text
Lịch chạy / yêu cầu người dùng
        ↓
Source registry + source adapter
        ↓
Phát hiện URL tài liệu mới
        ↓
Policy gate + hàng đợi ứng viên
        ↓
Fetch an toàn → trích xuất → kiểm tra metadata/chất lượng
        ↓
Checksum + version/deduplicate
        ↓
Chunk → FTS/vector embedding → ResearchStore
        ↓
RAG sử dụng tài liệu mới + báo cáo đồng bộ
```

Phạm vi đầu tiên chỉ là tài liệu nghiên cứu công khai: báo cáo tài chính, báo cáo thường niên, công bố thông tin, báo cáo kết quả kinh doanh và investor presentation. Dữ liệu giá thị trường vẫn do Market Provider quản lý, không trộn vào crawler tài liệu.

Hệ thống không vượt đăng nhập/paywall/CAPTCHA, không bỏ qua `robots.txt`, không dùng browser automation để né chặn và không mặc định coi một domain chính thức là giấy phép lưu toàn văn. Nếu quyền sử dụng chưa rõ, hệ thống chỉ lưu metadata, URL và đoạn trích ngắn phục vụ discovery; tài liệu được đưa vào hàng đợi duyệt thay vì tự ghi toàn văn vào RAG.

## 2. Hiện trạng có thể tái sử dụng

Các khối sau đã có và không cần viết lại:

- `SourceRegistry`: allowlist domain, publisher, quality tier, TTL và usage note.
- `SafeContentFetcher`: HTTPS, chống SSRF, giới hạn redirect, MIME và kích thước.
- `HtmlContentExtractor`: đọc HTML và PDF có text layer.
- `ResearchIngestionService`: TTL, quyền ingest, checksum, version và incremental ingest.
- `ResearchStore`: chunking, FTS5, vector embedding, hybrid retrieval và citation metadata.
- `WebResearchService`: Brave/search fallback cho discovery, nhưng hiện không lưu kết quả.

Các phần còn thiếu là catalog entrypoint, source-specific discovery, candidate queue, metadata validation, scheduler, lock/retry, audit report và cơ chế đưa kết quả tìm thấy trong chat vào pipeline lâu dài.

## 3. Kiến trúc mục tiêu

### 3.1 Thành phần mới

1. **ResearchSyncService**
   - Điều phối một lần đồng bộ.
   - Chọn nguồn đến hạn theo `refresh_hours`.
   - Áp ngân sách số trang, byte, thời gian và request/domain.
   - Không phải LLM Agent; đây là service deterministic.

2. **SourceAdapter**
   - Một interface chung: `discover(checkpoint) -> list[DocumentCandidate]`.
   - Adapter ưu tiên theo thứ tự: RSS/Atom → sitemap → trang danh sách IR → API JSON công khai → Brave `site:` fallback.
   - Mỗi website khác cấu trúc chỉ cần adapter/config riêng, không sửa toàn bộ crawler.

3. **CandidateRepository**
   - Lưu URL vừa phát hiện trước khi fetch/ingest.
   - Trạng thái: `DISCOVERED`, `FETCHED`, `QUARANTINED`, `INDEXED`, `UNCHANGED`, `RETRYABLE_ERROR`, `PERMANENT_ERROR`.
   - Cho phép chạy lại an toàn và xem vì sao một tài liệu bị loại.

4. **MetadataExtractor/Validator**
   - Trích title, publisher, ticker, document type, published time, reporting period và language.
   - Thứ tự tin cậy ngày công bố: metadata có timezone → ngày hiển thị trên trang/PDF → dữ liệu cấu trúc của nguồn. Không tìm được thì để trống và quarantine; tuyệt đối không dùng thời điểm crawl làm ngày công bố.
   - Rule/regex chạy trước. LLM chỉ được gọi khi metadata mơ hồ và phải trả schema; kết quả LLM mang trạng thái `NEEDS_REVIEW` nếu không có bằng chứng trong tài liệu.

5. **IngestionPolicyGate**
   - Tách rõ `discovery_allowed`, `fetch_allowed` và `storage_mode`.
   - `FULL_TEXT`: được phép lưu/chunk toàn văn.
   - `EXCERPT_ONLY`: chỉ lưu metadata và đoạn trích giới hạn.
   - `LINK_ONLY`: chỉ lưu URL/metadata, không đưa nội dung vào vector store.
   - `BLOCKED`: không fetch.

6. **SyncReporter**
   - Xuất JSON và bảng tóm tắt: URL mới, indexed, unchanged, quarantine, lỗi, số chunk/embedding, thời gian và request count.
   - Không ghi API key, prompt hay nội dung tài liệu vào log vận hành.

### 3.2 Dữ liệu trạng thái mới

Thêm các bảng SQLite additive, không phá dữ liệu cũ:

- `research_sources_state`: source id, checkpoint, lần chạy gần nhất, lần thành công gần nhất, lần chạy kế tiếp.
- `research_candidates`: canonical URL, parent/listing URL, discovered time, metadata, confidence, policy mode và status.
- `research_crawl_runs`: run id, start/end, source, counters, status và lỗi đã làm sạch.
- `research_crawl_failures`: URL, error type, retry count, next retry và permanent flag.
- `research_source_locks`: ngăn hai scheduler xử lý cùng nguồn đồng thời.

Không sao chép toàn văn vào candidate table; nội dung chỉ đi vào `ResearchStore` sau khi qua policy và validation.

## 4. Cấu hình nguồn

Mở rộng `config/research_sources.json` theo hướng tương thích ngược:

```json
{
  "source_id": "fpt-ir",
  "domain": "fpt.com.vn",
  "publisher": "FPT IR",
  "quality_tier": "T1",
  "entrypoints": [
    {
      "url": "https://fpt.com.vn/vi/nha-dau-tu/bao-cao-thuong-nien",
      "kind": "listing"
    }
  ],
  "allowed_path_prefixes": ["/vi/nha-dau-tu/"],
  "document_patterns": ["bao-cao", "annual-report", "financial-statement"],
  "ticker_map": ["FPT"],
  "discovery_allowed": true,
  "fetch_allowed": true,
  "storage_mode": "FULL_TEXT",
  "refresh_hours": 24,
  "min_request_interval_seconds": 3,
  "max_pages_per_run": 30,
  "usage_note": "Cần xác nhận điều kiện sử dụng trước khi bật FULL_TEXT."
}
```

`storage_mode` phải được cấu hình rõ cho từng nguồn. Giá trị mặc định khi migration là `LINK_ONLY`, không phải `FULL_TEXT`.

## 5. Các phase triển khai

### Phase C0 — Contract, state và policy gate

**Công việc**

- Thêm models: `SourceDefinition`, `SourceEntrypoint`, `DocumentCandidate`, `CrawlRun`, `IngestionDecision`.
- Mở rộng source registry nhưng giữ khả năng đọc cấu hình hiện tại.
- Thêm schema/migration cho bốn bảng trạng thái chính và lock.
- Thêm policy gate ba tầng discovery/fetch/storage.
- Bổ sung status `QUARANTINED` với reason code, không chỉ chuỗi lỗi.
- Chốt giới hạn mặc định: một worker, 3 giây/request/domain, 30 trang/source/run, 8 MB/tài liệu, 10 phút/run.

**File dự kiến**

- `models/research_crawler.py`
- `services/source_registry.py`
- `services/research_crawl_store.py`
- `services/ingestion_policy.py`
- `config/research_sources.json`

**Tiêu chí hoàn thành**

- Migration không làm mất corpus cũ.
- Nguồn chưa khai báo không thể auto-fetch/auto-ingest.
- Nguồn thiếu quyền toàn văn chỉ sinh `LINK_ONLY` hoặc `QUARANTINED`.
- Hai process không thể giữ cùng một source lock.

### Phase C1 — Discovery crawler cho nguồn chính thức

**Công việc**

- Xây interface `SourceAdapter` và bốn adapter miễn phí:
  - RSS/Atom adapter;
  - sitemap adapter;
  - HTML listing adapter;
  - Brave official-domain discovery adapter làm fallback.
- Chỉ theo link cùng allowlist domain/path, depth tối đa 1–2; không crawl toàn website.
- Chuẩn hóa URL trước khi ghi candidate; loại tracking URL và fragment.
- Lưu parent/listing URL để chứng minh tài liệu được phát hiện ở đâu.
- Dùng checkpoint và `ETag`/`Last-Modified` khi server hỗ trợ; nếu không, dùng checksum/listing fingerprint.
- Pilot 5 nguồn có giá trị demo: HOSE/HNX và ba issuer thường dùng. Chỉ thêm nguồn khi kiểm tra trực tiếp cấu trúc và điều kiện truy cập.

**File dự kiến**

- `services/research_discovery.py`
- `services/source_adapters/base.py`
- `services/source_adapters/rss.py`
- `services/source_adapters/sitemap.py`
- `services/source_adapters/html_listing.py`
- `services/source_adapters/brave_discovery.py`

**Tiêu chí hoàn thành**

- Chạy lại cùng checkpoint không tạo candidate trùng.
- Không đi ra ngoài domain/path đã cho phép kể cả qua redirect.
- Brave chỉ tạo candidate discovery; không tự biến snippet thành evidence.
- Trên bộ URL được gán nhãn thủ công của pilot, phát hiện ít nhất 90% tài liệu truy cập công khai thuộc loại đã cấu hình.

### Phase C2 — Fetch, metadata, quality gate và auto-ingestion

**Công việc**

- Nối candidate queue với `SafeContentFetcher` và `HtmlContentExtractor`.
- Mở rộng response fetch để giữ `ETag`, `Last-Modified`, kích thước và content type.
- Bổ sung parser metadata cho HTML/PDF; nhận diện ticker/type/period bằng rule trước.
- Kiểm tra tối thiểu trước ingest:
  - source/final URL hợp lệ;
  - title và publisher có thật;
  - `published_at` có timezone và có bằng chứng;
  - ticker/type hợp lệ hoặc được review;
  - nội dung không rỗng và không phải trang lỗi/login;
  - policy cho phép storage mode tương ứng.
- Gọi `ResearchIngestionService` để tận dụng version/checksum/dedup/chunk/embedding hiện có.
- Chỉ embed chunk mới; không reindex toàn corpus sau mỗi lần crawl.
- PDF scan được quarantine với `OCR_REQUIRED`, không coi là ingest thành công.

**File dự kiến**

- `services/research_sync.py`
- `services/document_metadata.py`
- `services/content_fetcher.py`
- `services/content_extractor.py`
- `services/research_ingestion.py`

**Tiêu chí hoàn thành**

- Tài liệu hợp lệ tự xuất hiện trong hybrid RAG sau một run.
- Run thứ hai với nội dung không đổi tạo 0 document/chunk/embedding mới.
- Cùng URL có nội dung sửa tạo revision và liên kết `supersedes_id`.
- Thiếu ngày công bố/quyền lưu/nội dung đọc được phải quarantine, không bịa metadata.

### Phase C3 — Scheduler, retry và vận hành trên ổ D

**Công việc**

- Thêm CLI:

```powershell
.\.venv\Scripts\python.exe scripts\sync_research_corpus.py --due
.\.venv\Scripts\python.exe scripts\sync_research_corpus.py --source fpt-ir --force-discovery
.\.venv\Scripts\python.exe scripts\review_research_candidates.py --status QUARANTINED
```

- Tạo script cài Windows Task Scheduler, mặc định mỗi ngày một lần và chạy bằng `.venv` trong `D:\Download\thayque\Agent`.
- Retry exponential có jitter cho timeout/429/5xx; 401/403/404 lặp lại chuyển permanent/long cooldown tùy loại.
- Dừng lịch sự khi hết budget, tiếp tục từ checkpoint ở lần sau.
- Sinh báo cáo tại `.runtime/research/reports/` và giữ log xoay vòng.
- Có health summary: nguồn nào quá hạn, lần thành công cuối, số quarantine và lỗi liên tiếp.

**Tiêu chí hoàn thành**

- Một nguồn lỗi không làm hỏng toàn run.
- Không vượt rate limit cấu hình của domain.
- Máy tắt/mất mạng không mất trạng thái; lần sau tiếp tục được.
- Mọi runtime data, model cache và report đều ở ổ D.

### Phase C4 — Lưu tài liệu phát hiện trong chat

**Công việc**

- Khi Research Agent tìm thấy URL mới, gọi `CandidateRepository.upsert()` thay vì chỉ trả URL rồi bỏ.
- Không ingest trực tiếp từ LLM output.
- Nếu URL thuộc registry, fetch được, metadata đầy đủ và policy cho phép, đẩy vào sync queue.
- Nếu thiếu điều kiện, UI hiện “Đã đưa vào hàng chờ duyệt” và lý do.
- Thêm màn hình Streamlit nhỏ: candidate mới, metadata trích xuất, policy, preview ngắn, nút Approve/Reject/Retry.
- Approval chỉ thay đổi candidate cụ thể; không tự bật toàn domain thành `FULL_TEXT`.

**Tiêu chí hoàn thành**

- Tài liệu tìm thấy trong chat tồn tại qua lần khởi động app tiếp theo.
- URL an toàn/đủ metadata có thể tự vào RAG; URL mơ hồ không được âm thầm ingest.
- Người dùng xem được nguồn, ngày, loại tài liệu và lý do quarantine trước khi duyệt.

### Phase C5 — OCR và nguồn động (tùy chọn, làm sau pilot)

**Công việc**

- OCR offline cho PDF scan bằng Tesseract/PaddleOCR chỉ khi thật sự cần; model/cache đặt trên D.
- Trích bảng chỉ bổ sung khi có use case định lượng rõ; không đưa bảng lỗi vào evidence.
- Browser automation chỉ dành cho trang JavaScript công khai không có RSS/API/HTML server-side và không dùng để vượt chặn.
- Nếu OCR confidence thấp, lưu tài liệu ở quarantine và không đưa số liệu vào claim.

**Tiêu chí hoàn thành**

- Đo OCR trên bộ PDF scan thật; số liệu quan trọng phải có kiểm tra riêng.
- Crawler thường vẫn hoạt động khi OCR/browser component không được cài.

## 6. Kiểm thử và đánh giá

### Unit/contract tests

- canonical URL, allowlist path, redirect, robots/policy và storage mode;
- parser RSS/sitemap/listing bằng fixture offline;
- metadata date/ticker/type/period và các trường hợp thiếu;
- queue transition, source lock, retry/backoff và checkpoint;
- exact duplicate, revision, same-content-different-URL;
- trang login/error giả không được ingest;
- PDF scan chuyển `OCR_REQUIRED`.

### Integration tests

- Dùng fake fetcher/source adapter; test không phụ thuộc Internet.
- Một run hoàn chỉnh: discover → candidate → fetch → extract → ingest → retrieve bằng RAG.
- Rerun phải idempotent.
- Một candidate độc hại/lỗi không chặn các candidate còn lại.

### Pilot acceptance report

Theo dõi riêng từng nguồn trong 14 ngày:

- discovery recall trên danh sách URL kiểm tra thủ công;
- tỷ lệ metadata bắt buộc đầy đủ;
- ingestion success trên tài liệu thực sự truy cập được;
- duplicate/re-embedding count;
- quarantine theo reason;
- request count, byte, duration P50/P95;
- độ trễ từ lúc nguồn công bố đến lúc có trong RAG;
- 20 câu hỏi thực tế kiểm tra retrieval và citation.

Không coi benchmark synthetic hiện tại là bằng chứng crawler hoạt động tốt trên website thật.

## 7. Definition of Done

Tính năng được xem là hoàn thành khi:

1. Ít nhất 5 nguồn chính thức pilot tự được kiểm tra theo lịch và có checkpoint.
2. Báo cáo mới hợp lệ tự đi từ discovery tới RAG mà không phải viết manifest thủ công.
3. Chạy lại không tạo duplicate hoặc embed lại nội dung không đổi.
4. Mọi document trong RAG có URL, publisher, published time, retrieved time, checksum, ticker/type và policy/usage note.
5. Không có URL ngoài allowlist hoặc nguồn `LINK_ONLY` bị lưu toàn văn.
6. Kết quả chat tìm thấy được lưu thành candidate và đi qua cùng policy gate.
7. Lỗi mạng, 429, PDF scan hoặc thay đổi HTML không làm app chat ngừng hoạt động.
8. Có report từng run và UI/CLI để xử lý quarantine.
9. RAG truy xuất được tài liệu mới và citation trỏ đúng document/version.
10. Toàn bộ dependency/cache/runtime nằm trên ổ D và không yêu cầu dịch vụ trả phí.

## 8. Thứ tự thực hiện đề xuất

Ưu tiên triển khai C0 → C1 → C2 trước. Khi một lần sync thủ công đã ổn định mới bật C3 scheduler; sau đó nối chat ở C4. C5 chỉ làm khi thống kê quarantine chứng minh PDF scan hoặc trang JavaScript là điểm nghẽn đáng kể.

MVP phù hợp để demo gồm 3 nguồn issuer + HOSE/HNX, HTML/PDF có text layer, một scheduler local và màn hình quarantine. Không cần Kafka, Airflow, Scrapy cluster, vector database riêng hay fine-tune LLM.

## 9. Giới hạn triển khai miễn phí cần chấp nhận

- Windows Task Scheduler chỉ chạy khi laptop có thể thực thi task và có mạng.
- Filesystem của Streamlit Community Cloud không phải nơi lưu corpus bền vững; crawler có thẩm quyền nên chạy ở máy local trên D. Bản deploy cần corpus snapshot được phép phân phối hoặc một storage bền vững riêng ở giai đoạn sau.
- Website có thể đổi cấu trúc, chặn IP cloud hoặc yêu cầu CAPTCHA; adapter phải báo lỗi và quarantine, không tìm cách vượt chặn.
- Brave free quota chỉ là fallback discovery, không dùng để quét hàng loạt mọi trang.
- E5 cải thiện retrieval nhưng không bắt buộc cho crawler; FTS vẫn hoạt động nếu chưa cài model.
- LLM không xác nhận bản quyền và không phải nguồn sự thật cho ngày công bố/ticker/số liệu.
