# Frozen coordination contract benchmark

Chạy từ workspace bằng Python trong `.venv`:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_coordination.py
```

Report mặc định: `.runtime/evaluations/coordination-latest.json` trên ổ D.
Không tải market data, gọi LLM hoặc đọc corpus nghiên cứu cá nhân. Graph dùng
`InMemorySaver` riêng cho mỗi lần chạy, không ghi analysis checkpoint sản phẩm.

Ba biến thể nhận deep copy của cùng snapshot và cùng `RequestBudget`:

| Variant | Cơ chế |
|---|---|
| `single_pass` | Full production graph, `max_iterations=0`, vẫn có sáu vai trò và Risk |
| `bounded_coordination` | Cùng graph, giới hạn mặc định hai vòng tái phân tích |
| `no_risk_baseline` | Chạy Data/Research/Technical/Fundamental, tổng hợp hai signal bằng quy tắc riêng chỉ trong evaluator; không gọi product Manager |

Dataset `synthetic-frozen-coordination-v1` có bốn snapshot tổng hợp với ngày cố
định: dữ liệu đầy đủ, thiếu fundamentals, P/E mâu thuẫn cùng kỳ/phạm vi và sự kiện
đình chỉ giao dịch. Đây là dữ liệu tác giả tạo để kiểm tra hợp đồng, không phải
lịch sử giá thật hoặc nhãn chuyên gia độc lập. Snapshot hash xác nhận input giống
nhau giữa các biến thể. Ngày và `quality_report.checked_at` đều cố định.

Report đo vi phạm constraint, số signal `UNKNOWN`, kết quả còn thiếu, conflict chưa
giải quyết, số vòng, latency và tool/LLM/API calls. Conflict của hai graph được đo
bằng production typed conflict detector, không phải oracle độc lập; no-risk báo
`null` cho conflict chưa được critic đánh giá. Tool count chỉ tính các ranh giới
đã instrument, không phải mọi phép tính Python. Latency phụ thuộc máy và scheduling.

Benchmark không mặc định coi nhiều agent hơn là tốt hơn: tái phân tích có thể tốn
thêm call mà vẫn giữ nguyên số conflict khi input đóng băng. Không suy ra lợi
nhuận, calibration, độ chính xác đầu tư hoặc ưu thế tổng quát từ các kết quả này.

Có thể cung cấp `--dataset path.json` với object `{"cases": [...]}`. Mỗi case
tuân theo `CoordinationCase`: `case_id`, `market_data` theo `MarketData`, horizon,
`forbidden_recommendations`, lý do constraint và `label_provenance`. Nhãn
`HUMAN_REVIEWED_SNAPSHOT_CONSTRAINT` là lời khai của người tạo dataset, service
không tự xác nhận đã có human review. Không để bí mật hoặc dữ liệu cá nhân vào
dataset/report. Thêm nhãn độc lập và snapshot point-in-time cần thiết trước khi
dùng benchmark cho quyết định triển khai.
