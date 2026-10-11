# Phase C — Research, Retrieval & Evidence

## Trạng thái

Foundation đã triển khai theo hướng citation-first. Research chạy **hybrid theo tầng**: SQLite FTS5 tìm trong corpus cục bộ trước; local miss chạy một research vertical slice có giới hạn: plan → source discovery → fetch HTML an toàn → extract → evidence selection. Chưa thêm vector database/LLM summarizer vì chưa có benchmark chứng minh lợi ích.

## Contracts và storage

`models/research.py` định nghĩa `ResearchDocument`, `DocumentChunk`, `Citation`, `WebCitation`, `ResearchPlan`, `FetchedDocument`, `ExtractedDocument`, `ExternalEvidenceCitation`, `EvidenceClaim` và `EvidenceBundle`.

Mỗi document bắt buộc có:

- title, type, publisher;
- URL nguồn có thể kiểm tra;
- `published_at` timezone-aware và local `retrieved_at`;
- SHA-256 content checksum, ticker/language;
- chunks có offset và citation gắn tới chính xác document/chunk.

`ResearchStore` lưu metadata/chunk trong `.runtime/research/research.sqlite3` trên ổ D và dùng SQLite FTS5 lexical retrieval. Đây là lựa chọn cố ý: dữ liệu ban đầu nhỏ, inspect/debug dễ, không cần embedding model hoặc vector database để trả citation đúng.

## Ingestion và retrieval

Hiện hỗ trợ UTF-8 `.txt`, `.md`, `.html`, `.htm`. HTML được bỏ tag/script/style trước khi chunk. PDF/OCR chưa được hỗ trợ vì chưa cài parser và không được giả rằng file PDF đã đọc được.

Ingest một tài liệu đã được phép dùng:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_research_document.py D:\documents\fpt-release.html `
  --title "FPT earnings release" `
  --type earnings_release `
  --source-url "https://example.com/original-document" `
  --publisher "FPT IR" `
  --published-at "2026-10-01T09:00:00+07:00" `
  --ticker FPT
```

Search verified documents:

```powershell
.\.venv\Scripts\python.exe scripts\search_research.py "tăng trưởng doanh thu" --ticker FPT
```

Với một trang HTML official đã được duyệt, có thể fetch và ingest trực tiếp; script từ chối non-HTML và giới hạn payload 3 MB:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_research_url.py `
  --title "Official disclosure" --type regulatory_disclosure `
  --source-url "https://issuer.example/disclosure" --publisher "Issuer IR" `
  --published-at "2026-10-01T09:00:00+07:00" --ticker FPT
```

Chat query có các dấu hiệu như “báo cáo”, “công bố thông tin”, “filing”, “nghị quyết” route tới `RESEARCH`.

### Hybrid retrieval

`ChatService` luôn gọi `ResearchStore` trước. Local result có chunk quote, publisher, `published_at`, local `retrieved_at` và checksum nên được hiển thị là **Verified local citations**. Local FTS có một số alias deterministic Việt–Anh cho báo cáo/công bố/nghị quyết để document English vẫn được tìm bằng câu hỏi Việt. Nếu không có local result và có ticker, `WebResearchService` thử trang hồ sơ/công bố chính thức HNX theo mã; đây tránh lỗi mã ngắn như `PNG` bị search engine hiểu là định dạng ảnh. Kết quả HNX không chặn các nguồn bổ sung: service vẫn chạy chuỗi web miễn phí để tìm tin tức, IR và bối cảnh ngành, rồi gộp/deduplicate theo URL. Khi không có hồ sơ HNX, service lấy tên doanh nghiệp từ VCI/vnstock để khử mơ hồ. Truy vấn luôn giữ nguyên chủ đề người dùng nêu (ví dụ hoạt động, mở rộng, rủi ro), thay vì biến mọi câu hỏi thành một truy vấn mã cổ phiếu chung. Web discovery hiển thị **Live web results (not verified evidence)** gồm URL đích, domain, search provider, snippet và UTC `retrieved_at`.

Web result không có `published_at`, không được ghi vào SQLite, không được dùng để tạo claim/tóm tắt và không thay thế tài liệu đã xác minh. Người dùng cần mở URL, kiểm tra quyền dùng/source/timestamp, sau đó ingest tài liệu primary đã duyệt nếu muốn nó trở thành evidence offline. Điều này ngăn timestamp của lúc tìm kiếm bị nhầm thành ngày công bố.

### External evidence vertical slice

Khi web discovery trả URL, `ResearchOrchestrator` có thể fetch tối đa số trang đã cấu hình trong một request. `SafeContentFetcher` chỉ cho HTTPS public host, chặn URL có credentials/local/private IP, kiểm tra từng redirect, giới hạn byte và timeout, rồi chỉ nhận HTML/XHTML. `HtmlContentExtractor` bỏ script/style/navigation/form, lấy title, canonical URL, ISO publication time nếu source cung cấp timezone, text và SHA-256.

Evidence external có quote từ **body đã fetch**, canonical URL, retrieval time, publication time nếu có, hash, source domain và relevance score. Nó được hiển thị là **Extracted external evidence** và chỉ sống trong response metadata; không tự động được ghi vào `research.sqlite3` hay mang nhãn verified local. Nếu fetch/extract thất bại, UI hạ xuống **Live web results (not verified evidence)** hoặc abstain.

Research plan hiện deterministic, tạo tối đa ba complementary query cho ý định báo cáo/outlook. Budget được cấu hình qua `.env`:

```text
STOCK_AGENT_RESEARCH_MAX_SEARCHES=3
STOCK_AGENT_RESEARCH_MAX_PAGES=3
STOCK_AGENT_RESEARCH_MAX_ITERATIONS=1
STOCK_AGENT_RESEARCH_TIMEOUT_SECONDS=20
STOCK_AGENT_RESEARCH_MAX_TOTAL_BYTES=3000000
```

Chế độ `free` (mặc định) không cần API key: GDELT DOC 2.0 discovery cho tin tức trước, sau đó Bing RSS compatibility fallback. Trang hồ sơ/công bố HNX và các trang IR doanh nghiệp do web discovery tìm được vẫn được giữ cùng các nguồn bổ sung. Đây là discovery miễn phí, không phải cam kết search quality/SLA. Research planner tạo tối đa ba truy vấn: câu hỏi nguyên gốc và hai truy vấn bổ trợ về vận hành/kết quả kinh doanh, hoặc rủi ro/bối cảnh ngành. Relevance gate yêu cầu nguồn web tổng quát phải nêu đúng ticker, tránh lẫn báo cáo của doanh nghiệp khác; mọi URL chỉ trở thành evidence sau khi fetch và extract HTML thành công. Khi có `BRAVE_SEARCH_API_KEY`, `auto` hoặc `brave` dùng Brave Search API chính thức (structured JSON, country/language/freshness config). Có thể tắt toàn bộ network fallback:

```powershell
$env:STOCK_AGENT_WEB_SEARCH_ENABLED = "false"
```

Hoặc cấu hình dài hạn qua `.env`:

```text
STOCK_AGENT_WEB_SEARCH_ENABLED=true
STOCK_AGENT_WEB_SEARCH_PROVIDER=free
BRAVE_SEARCH_API_KEY=
STOCK_AGENT_WEB_SEARCH_FRESHNESS=
STOCK_AGENT_OFFICIAL_ISSUER_LOOKUP_ENABLED=true
STOCK_AGENT_COMPANY_NAME_LOOKUP_ENABLED=true
```

## Corpus đã ingest

Hai trang HTML trên website Báo cáo thường niên 2022 chính thức của FPT Corporation đã được ingest cho ticker `FPT`. Mốc công bố `2023-03-27T00:00:00+07:00` là ngày trên tài liệu Báo cáo thường niên 2022 của FPT; checksum dưới đây xác định đúng version HTML đã fetch.

| Title | Source | SHA-256 prefix |
|---|---|---|
| FPT 2022 Annual Report – Sustainable construction | [official page](https://bctn2022.fpt.com.vn/en/highlight/sustainable-construction) | `b4c7ca23dff2` |
| FPT 2022 Annual Report – 2023 planning and direction | [official page](https://bctn2022.fpt.com.vn/en/strategy/planning-and-direction) | `d1cc1fea67b3` |
| Hoa Phat annual report catalogue – Annual Report 2024 | [official IR page](https://www.hoaphat.com.vn/investor-relations/annual-report) | `3a76d56285e1` |
| Hoa Phat information disclosure catalogue | [official disclosure page](https://www.hoaphat.com.vn/investor-relations/information-disclosure) | `42c55b62600a` |
| Vinamilk sustainability report catalogue – 2024 | [official IR page](https://i04-eshop-vn.apps.vinamilk.tech/en/investor/reports/sustainability) | `baca5f3aadf3` |
| Phat Dat annual report catalogue – Annual Report 2024 | [official IR page](https://www.phatdat.com.vn/en/investor-relations?tab=annual-reports) | `2c40d2f67922` |
| HNX periodic disclosure index – PNG Annual Report 2025 | [official HNX page](https://www.hnx.vn/vi-vn/cophieu-etfs/chi-tiet-chung-khoan-uc-PNG.html) | `66a01e7e1223` |

## Giới hạn còn mở

- Production-grade credentialed search provider/failover, source allowlist, robust main-content extraction, source-quality score, retention cache và freshness SLA chưa có.
- Chưa có PDF parsing, OCR, table extraction hoặc document version/revision workflow đầy đủ.
- Local retrieval là lexical FTS; web fallback không phải vector retrieval. Chỉ đánh giá embedding/reranker sau khi có corpus thật và metric recall/citation faithfulness.
- Chưa có multi-iteration recovery, cross-source contradiction handling hoặc LLM claim synthesis. Khi bổ sung, mọi claim phải link citation và numerical facts vẫn phải lấy từ structured data.
