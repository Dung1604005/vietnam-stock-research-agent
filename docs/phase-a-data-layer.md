# Phase A — Reliable Data Layer

> Đối chiếu code 23/10/2026: cập nhật default pacing/retry/deadline và scoring bên dưới. Smoke FPT 09/10 là bằng chứng lịch sử, chưa gọi lại upstream trong lượt audit tài liệu.

## Trạng thái

Foundation của Phase A đã được triển khai. Production gates còn mở được liệt kê ở cuối tài liệu; hệ thống chưa được tuyên bố production-ready.

## Data contracts

`models/data.py` bổ sung:

- `Instrument`: ticker, type, currency và các metadata có thể biết chắc. Exchange/name để `None` thay vì đoán.
- `SourceReference`: provider/source ID, data kind, `as_of`, timezone-aware `fetched_at`, request parameters, schema version, SHA-256 và snapshot ID.
- `FinancialMetricObservation`: value, unit, reporting-period label/end, published time nếu nguồn có cung cấp và source ID.
- `PriceAdjustment`: `ADJUSTED`, `UNADJUSTED` hoặc `UNKNOWN`.
- `CorporateAction`: contract đã có; provider hiện chưa cung cấp action nên không tạo dữ liệu giả.
- `DataQualityReport`/`DataQualityIssue`: PASS/WARN/FAIL và các check đã chạy.

`MarketData` vẫn giữ các field cũ để tương thích agents, đồng thời mang instrument, observations, provenance, quality, adjustment policy và schema version.

## Acquisition và provenance

Production provider hiện là vnstock/VCI. Mỗi fetch tạo source reference riêng cho:

- `VCI:Quote.history:<ticker>` — OHLCV as-of phiên cuối.
- `VCI:Finance.ratio:<ticker>` — financial ratios as-of kỳ báo cáo lấy trực tiếp từ payload.

Không suy đoán `published_at`, exchange, source URL hoặc adjusted-price status khi provider không trả field đó.

Raw payload được lưu bất biến, gzip và đánh địa chỉ bằng SHA-256 tại:

```text
.runtime/snapshots/<provider>/<fetch-date>/<sha256>.json.gz
```

Response/UI hiển thị data as-of, fetched timestamp, source ID, schema và snapshot checksum. `.runtime` được ignore khỏi git và bắt buộc nằm trên ổ D trong workspace Windows này.

## Cache và resilience

Normalized `MarketData` được cache trong SQLite:

```text
.runtime/cache/market_data.sqlite3
```

Cache key chứa normalizer schema version, source, ticker, lookback và việc có yêu cầu financials hay không. Payload cache luôn được chạy lại data-quality checks để freshness không bị đóng băng.

Default policy:

- Cache TTL mặc định: 259200 giây (3 ngày). Recommendation chỉ force-refresh các mã lọt vào phân tích sâu; không tải lại toàn bộ universe trong mỗi prompt.
- Provider deadline là cooperative; timeout chỉ được clamp khi method hỗ trợ. `VNSTOCK_TIMEOUT` không bảo đảm mọi vnstock explorer call bị ngắt sau 20 giây.
- VnstockMarketDataProvider mặc định 1 attempt, cấu hình qua `STOCK_AGENT_PROVIDER_MAX_ATTEMPTS`; nếu bật retry thì backoff từ 0.4 giây. Generic ResiliencePolicy có default 3 nhưng provider ghi đè.
- Provider mặc định pace mỗi attempt 3.2 giây khi chưa có API key hoặc 1.05 giây khi có key; có thể cấu hình riêng. Generic ResiliencePolicy 0.25 giây không phải default thực tế của provider.
- Vnstock internal retries: tắt để tránh retry lồng nhau.
- Circuit breaker: mở sau 5 failed operations, cooldown 60 giây.

Tất cả đều cấu hình được qua `.env.example`. Provider giữ in-process health telemetry (cache hit/miss, attempts, retries, circuit state, latest success/error và cumulative rate-limit wait); `scripts/data_quality_report.py --show-health` in telemetry này sau khi chạy. Counters reset khi process restart; durable metrics/tracing thuộc production-operations scope, không bị giả là đã có.

## Data-quality policy

Các check hiện có:

- Minimum price history.
- Unique ascending dates.
- Future/stale as-of.
- Large business-day gaps.
- Extreme returns có thể do corporate actions chưa xử lý.
- Zero-volume coverage.
- Structured source reference bắt buộc.
- Unknown adjusted-price status.
- Financial completeness, reporting period và stale period.

Lỗi provenance/chronology/future date làm toàn dataset `FAIL` và provider không trả analysis. Warning vẫn cho phép route không liên quan sử dụng phần data hợp lệ.

Financial scoring hiện đánh giá coverage/quality của cả ratio và statement: statement còn mới/tương thích được ưu tiên, ratio cũ có thể giảm trọng số, không có metric/statement dùng được thì UNKNOWN. Không mô tả một stale flag là lệnh chặn mọi dữ liệu tài chính. Price/technical route vẫn có thể chạy.

## Phát hiện từ dữ liệu thật

Smoke test FPT ngày 09/10/2026 cho thấy:

- OHLCV: 396 phiên, as-of 09/10/2026.
- VCI financial response đang gắn các ratios khả dụng với kỳ 2018.
- Quality: `WARN` với `PRICE_GAPS`, `UNKNOWN_PRICE_ADJUSTMENT`, `STALE_FINANCIALS`.
- Fundamental Engine: `UNKNOWN`, confidence 0.1; không chấm điểm từ ratios 2018.

Raw snapshot chứng minh kỳ 2018 đến từ payload upstream. Code giữ nguyên provenance đó, không tự đổi thành kỳ mới.

## Operations

Chạy quality report cho một số mã:

```powershell
.\.venv\Scripts\python.exe scripts\data_quality_report.py FPT VNM HPG
```

Chạy toàn configured universe:

```powershell
.\.venv\Scripts\python.exe scripts\data_quality_report.py
```

Chỉ kiểm tra price:

```powershell
.\.venv\Scripts\python.exe scripts\data_quality_report.py FPT --prices-only
```

In cache/resilience health của lần chạy:

```powershell
.\.venv\Scripts\python.exe scripts\data_quality_report.py FPT --show-health
```

## Production gates còn mở

- Chọn và phê duyệt nguồn thứ hai/nguồn chính thức cho reconciliation, nhất là financial statements.
- Xác định source licensing và SLA.
- Tích hợp lịch giao dịch Việt Nam để phân biệt nghỉ lễ với missing bars.
- Xác nhận adjusted/unadjusted semantics hoặc thêm corporate-action feed.
- Bổ sung instrument master có exchange/listing status và point-in-time history.
- Chạy quality report/contract smoke định kỳ đủ dài để đặt SLO thực tế.

Cho tới khi đóng các gate này, source warnings phải được giữ nguyên và Fundamental phải abstain khi period provenance không đạt yêu cầu.
