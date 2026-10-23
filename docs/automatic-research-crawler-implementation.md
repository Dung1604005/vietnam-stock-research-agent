# Crawler và auto-ingestion — triển khai và sử dụng

> Phạm vi báo cáo: số test, latency, corpus và smoke test ghi trong tài liệu là kết quả của lượt triển khai lịch sử, không phải tổng mới ngày 23/10/2026. Trạng thái/code và kiểm chứng mới được tổng hợp tại [project-report.md](project-report.md).

Pipeline C0–C4 đã có code: discovery theo nguồn → hàng đợi SQLite → policy/metadata → ingest tăng dần → FTS/vector RAG. Script hỗ trợ cài Windows Task Scheduler tên `VietnamStockResearchSync`, mặc định **09:00 giờ máy**, dùng `.venv\Scripts\pythonw.exe` trên ổ D. Lượt triển khai trước đã ghi nhận cài task; audit code 23/10/2026 không xác minh lại trạng thái task trên máy. Task dùng tài khoản hiện tại và chỉ chạy khi tài khoản có phiên đăng nhập; máy cần có mạng. Nếu bỏ lỡ lịch, `StartWhenAvailable` cho phép chạy bù.

Kiểm tra năm/quý, độ mới và lựa chọn tài liệu hiện tại/lịch sử: [research-data-freshness.md](research-data-freshness.md).

## Cách dùng nhanh trên app

1. Chạy app bằng lệnh dưới đây.
2. Ở sidebar, mở **Thư viện báo cáo / Crawler**.
3. Chọn nguồn, bấm **Kiểm tra báo cáo mới** nếu muốn kiểm tra trước lịch hằng ngày.
4. Xem trạng thái `QUARANTINED`: hệ thống ghi rõ lý do, URL nguồn và metadata.
5. Với tài liệu được phép dùng, nhập cơ sở sử dụng, sửa các metadata còn thiếu nếu cần, chọn **Duyệt lưu toàn văn tài liệu này** rồi **Lưu**.
6. Bấm **Ingest các tài liệu đã duyệt**, hoặc chờ lịch chạy tiếp theo. Khi trạng thái là `INDEXED`, có thể hỏi đáp tài liệu bằng RAG.

```powershell
Set-Location "D:\Download\thayque\Agent"
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Các URL do research tìm thấy trong chat cũng tự được ghi vào cùng thư viện nếu thuộc nguồn/path đã cấu hình. Chat chỉ ghi metadata, không ghi search snippet thành nội dung tài liệu. Crawler chạy riêng nên không thêm một lượt crawl dài vào mỗi câu hỏi.

## Chế độ tự động

Có hai mức:

- **Theo tài liệu:** nguồn giữ `LINK_ONLY`; bạn duyệt từng báo cáo trên UI một lần, worker tự fetch/kiểm tra/ingest sau đó. Các phiên bản sửa tại cùng URL tiếp tục được kiểm tra theo TTL.
- **Theo nguồn đã được phép:** sau khi xác nhận quyền lưu/chunk tài liệu của nguồn, cấu hình `storage_mode: "FULL_TEXT"` và ghi cơ sở đó trong `usage_note` của nguồn tương ứng ở `config/research_sources.json`. Tài liệu mới đủ metadata từ nguồn này được tự ingest, không cần tạo manifest hay duyệt mỗi báo cáo. Các mục trước đó chỉ bị giữ với `PERMISSION_REQUIRED` cũng tự được xử lý lại khi policy nguồn chuyển sang `FULL_TEXT`.

Các nguồn pilot được đặt `LINK_ONLY` vì domain chính thức không tự xác nhận quyền tái sử dụng toàn văn. Chưa có bản full-text nào từ pilot được tự động cấp phép hoặc ingest trong lần triển khai này. Kho RAG hiện có không bị xóa.

`EXCERPT_ONLY` cho phép lưu preview tối đa 600 ký tự trong hàng đợi, không index toàn văn. `BLOCKED` không được fetch. Duyệt một tài liệu không thay đổi cấu hình toàn domain.

## Lệnh vận hành

Kiểm tra các nguồn đến hạn và xử lý hàng chờ:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\sync_research_corpus.py --due
```

Kiểm tra một nguồn ngay; ngân sách nhỏ phù hợp để thử:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\sync_research_corpus.py --source fpt-ir --force-discovery --seconds 30 --pages 6
```

Chỉ xử lý các tài liệu đã duyệt/chờ xử lý, không đi qua trang danh sách:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\sync_research_corpus.py --queue-only
```

Xem sức khỏe nguồn và danh sách cần duyệt:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\sync_research_corpus.py --health
.\.venv\Scripts\python.exe -X utf8 scripts\review_research_candidates.py --status QUARANTINED --limit 20
```

CLI cũng hỗ trợ `--candidate ID --action approve --note "cơ sở quyền sử dụng"`, `--published-at`, `--tickers`, `--document-type`; ngày công bố nhập ISO-8601 có múi giờ, lấy từ nguồn. UI giúp tránh phải tự viết manifest/CLI dài.

## Lịch chạy Windows

Kiểm tra task có tồn tại và đường dẫn còn đúng trước khi chạy hoặc quyết định cài lại:

```powershell
Get-ScheduledTask -TaskName VietnamStockResearchSync
Get-ScheduledTaskInfo -TaskName VietnamStockResearchSync
Start-ScheduledTask -TaskName VietnamStockResearchSync
```

Ở máy khác hoặc khi tạo lại môi trường:

```powershell
.\scripts\install_research_schedule.ps1 -At "09:00"
```

Gỡ đúng task của tính năng này:

```powershell
.\scripts\install_research_schedule.ps1 -Remove
```

Script không ghi đè một task cùng tên đã tồn tại. `pythonw.exe` chạy nền không bật cửa sổ terminal. Không lưu password tài khoản vào task. Nếu di chuyển repository, cần gỡ/cài lại task để cập nhật đường dẫn.

## Báo cáo và nơi lưu dữ liệu

- Hàng đợi, state, source lock, lịch sử run, lỗi: các bảng `research_*` trong `.runtime/research/research.sqlite3`.
- Nội dung đã qua gate: `research_documents`, `research_chunks`, `research_chunk_embeddings` trong cùng SQLite.
- Báo cáo từng run: `.runtime/research/reports/<timestamp>-<run_id>.json`, giữ tối đa 100 báo cáo do crawler tạo.
- Tất cả các đường dẫn trên nằm trong workspace ổ D; `.runtime/` đã được Git ignore.

Nếu có cấu hình `STOCK_AGENT_RUNTIME_DIR`, dùng thư mục `research` dưới runtime đó, vẫn phải trên D ở Windows.

Ý nghĩa trạng thái:

| Trạng thái | Ý nghĩa |
|---|---|
| `DISCOVERED` | Đã lưu URL và chờ worker |
| `FETCHED` | Đã tải, đang xử lý metadata/nội dung |
| `INDEXED` | Đã vào RAG |
| `UNCHANGED` | Đã kiểm tra, nội dung không đổi; không tạo lại embedding |
| `QUARANTINED` | Cần bổ sung metadata, quyền lưu, OCR hoặc xử lý nguồn |
| `RETRYABLE_ERROR` | Lỗi tạm thời; có thời điểm retry và exponential backoff |
| `PERMANENT_ERROR` | HTTP không thể retry hợp lý hoặc đã hết số lần thử; cần thao tác Retry |
| `REJECTED` | Người dùng loại hoặc URL là danh mục thay vì tài liệu |

Report có request count, byte tải, số candidate/document/chunk/vector mới và trạng thái từng nguồn. `PAUSED_BUDGET` giữ checkpoint để tiếp tục; không phải hoàn thành toàn bộ nguồn. `COMPLETED` là hoàn thành lượt chạy trong phạm vi hiện tại; không có nghĩa mọi tài liệu đã được cấp phép/index.

Deadline được kiểm tra giữa các request, với timeout HTTP tối đa 12 giây mỗi lần. Đây là ngân sách điều phối; HTTP/DNS đang chạy không bị hủy cưỡng bức đúng tại giây cuối. Hai nút đồng bộ trên UI dùng ngân sách 60 giây; CLI/lịch chạy mặc định 600 giây.

## Cấu trúc code

- `models/research_crawler.py`: source, candidate, status, policy/run contracts.
- `services/research_crawl_store.py`: queue, review, state/checkpoint, lock và health.
- `services/research_crawl_transport.py`: robots, HTTPS/SSRF, domain/path/redirect, pacing và budget.
- `services/source_adapters/`: RSS/Atom, sitemap, HTML, JSON, FPT download normalization và Brave fallback.
- `services/research_discovery.py`: discovery BFS, canonical URL, pagination, frontier và HTTP validators.
- `services/document_metadata.py`: ngày công bố, ticker/type/period, chất lượng nội dung.
- `services/ingestion_policy.py`: quyền lưu theo source và review từng tài liệu.
- `services/research_sync.py`: điều phối, retry, incremental ingest, report.
- `services/research_ingestion.py`: dùng nội dung đã fetch, tránh tải lại tài liệu trong cùng run.
- `ui/research_library.py`: thư viện và thao tác duyệt.

Không thêm dependency crawler mới. HTML/XML dùng standard library; PDF text dùng `pypdf` đã có. E5 là tùy chọn và không được tự tải/cài thêm ở lượt này.

## Kết quả pilot và giới hạn còn lại

Pilot thực tế ngày 17/10/2026 đã thử năm nguồn: `fpt-ir`, `hpg-ir`, `mwg-ir`, `hnx-disclosures`, `hose-disclosures`.

- FPT: phát hiện tài liệu công bố qua `/api/media/*.pdf`; gộp nút download và view thành một URL. Trang công bố cung cấp ngày theo từng mục.
- Hòa Phát: nhận diện PDF trên `file.hoaphat.com.vn/hoaphat-com-vn/`; ngày công bố lấy từ đúng hàng listing. Có 15 URL PDF mới trong lượt pilot ngân sách nhỏ.
- MWG: trang hiện tại `/` và `/bao-cao`; HTML nhận được chủ yếu là navigation, nội dung báo cáo cần adapter JSON public cụ thể hoặc browser tùy chọn. Không coi danh mục là tài liệu.
- HNX: truy cập HTTPS thất bại ở môi trường pilot; trạng thái lỗi được ghi nhận, không được tính là crawl thành công.
- HOSE: HTTP trả HTML nhưng chưa có link tài liệu trong nội dung nhận được. Cần adapter endpoint công khai hoặc discovery fallback đã cấu hình.

Brave adapter có sẵn nhưng `brave_fallback` mặc định tắt; nếu bật theo source và đã cấu hình key, nó chỉ tìm URL thuộc allowlist, không dùng snippet làm evidence. Không tự tạo tài khoản/gói trả phí.

Đã kiểm thử bằng fixture độc lập với Internet: end-to-end vào RAG, idempotency và không embed lại, version sửa đổi, ngày công bố bản mới, quyền nguồn/review, lock, retry, redirect/path, robots, RSS/Atom/sitemap/JSON, FPT/CDN, checkpoint khi hết ngân sách, PDF scan và cô lập lỗi từng tài liệu. Fixture không đi vào corpus thật.

Regression toàn repository: **363 tests passed**. Smoke-test giao diện thư viện trên dữ liệu hàng chờ thực tế: **0 exception**, hiển thị 50 form duyệt; không gọi market API/LLM trong smoke-test này.

C5 (OCR và browser) vẫn là nhánh tùy chọn theo kế hoạch; chưa cài engine/model/browser mới. PDF không có text layer được giữ với `OCR_REQUIRED`. Các website yêu cầu JavaScript/chặn truy cập được báo rõ thay vì tự vượt chặn.

Các tiêu chí cần thời gian/dữ liệu ngoài lượt triển khai vẫn chưa xác nhận: theo dõi 14 ngày, discovery recall ≥90% trên URL gán nhãn thủ công, đủ năm nguồn hoạt động ổn định, 20 Q&A trên corpus báo cáo được phép dùng và latency mạng P50/P95. Không thay các tiêu chí này bằng kết quả test offline.
