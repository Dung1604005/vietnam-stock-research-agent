# Multi-agent reliability và giới hạn vận hành

Cập nhật đối chiếu 23/10/2026. Sáu vai trò của ứng dụng vẫn dùng supervisor/shared state;
các Codex hỗ trợ sửa repository không phải agent chạy trong sản phẩm.

## Quyết định và bằng chứng

Manager chặn `BUY` khi còn conflict `ERROR` chưa giải quyết thuộc dữ liệu, thời
gian, metric hoặc policy; bất đồng diễn giải riêng lẻ không tự động trở thành lỗi
dữ liệu. Quan sát tài chính chỉ so sánh trong cùng kỳ, đơn vị, loại statement,
basis, consolidation, currency và scale đã khai báo. Risk veto và độ bất định dữ
liệu vẫn là đầu vào authoritative.

Confidence tổng hợp dùng confidence của Technical, Fundamental và Risk, sau đó
giảm theo conflict/evidence coverage. Mức risk thấp không đại diện cho độ chắc
chắn cao. Các confidence này chưa calibrated và không phải xác suất sinh lợi.

Evaluator tách execution coverage khỏi analytical sufficiency. Báo cáo đầy đủ
nhưng signal `UNKNOWN`, evidence không có, `UNAVAILABLE`, confidence thấp hoặc
conflict dữ liệu còn mở không được coi là phân tích thành công.

## Hợp đồng thực thi và lỗi cục bộ

Task execution ghi `run_id`, `task_id`, input fingerprint, node, trạng thái và
output keys. Message mang correlation với ngữ cảnh task; runtime kiểm tra output
update, sender và audit envelope theo phạm vi vai trò. Tool selection của LLM
dùng allowlist riêng; `AgentSpec.allowed_tools` không phải sandbox cấp quyền cho
mọi lời gọi Python. Các contract trong cùng tiến trình không cô lập mã không tin cậy.

Graph giữ kết quả các bước thành công khi specialist hoặc optional stage lỗi.
Specialist không khả dụng trả typed output bảo thủ và execution report, để
Manager có thể hoàn tất báo cáo thiếu dữ liệu. Không tạo số đo thị trường thay
thế. Lần lấy dữ liệu ban đầu vẫn cần dữ liệu hợp lệ; refresh thất bại giữ snapshot
đã có và ghi nguyên nhân. Kết quả partial có thể còn conflict chưa giải quyết.

Routing plan được kiểm tra dependency/input/output và ngân sách trước thực thi.
Executor chạy task đã validate; full-analysis task group dùng supervisor graph.
Routing template tiếp tục là allowlist cố định, không phải planner tự phát sinh
code hoặc công cụ tùy ý.

## Deadline và retry

`RequestMetrics.remaining_seconds()` và `check_deadline()` áp request budget;
`task_deadline_scope()` thu hẹp deadline cho từng task. Deadline được kiểm tra
trước công việc có ngân sách, trước/sau pacing và từng attempt/retry. Backoff bị
giới hạn bởi thời gian còn lại. Hết ngân sách không tính là outage của upstream
và không retry hoặc mở circuit.

Transport hỗ trợ timeout được clamp vào thời gian còn lại. vnstock không expose
timeout cho mọi explorer method, nên một synchronous call đang chạy có thể trở
về muộn. Hệ thống dừng hợp tác tại ranh giới kế tiếp; không bảo đảm hard wall
clock hoặc có thể hủy CPU/native I/O tùy ý. Không có worker bị bỏ lại chỉ để trả
timeout sớm. Finalization, lưu partial và telemetry vẫn được phép hoàn tất.

Search-provider fallback, tra tên doanh nghiệp, từng redirect, DNS và embedding
đều được kiểm tra tại ranh giới gọi; timeout HTTP được thu hẹp theo thời gian còn
lại. Kết quả specialist trả về sau deadline được ghi `UNAVAILABLE`, không được
dùng để cấp kết luận `BUY` mới.

## Checkpoint và tiếp tục

Analysis graph lưu LangGraph checkpoint và pending writes vào SQLite local
trong `.runtime/runs/analysis.sqlite3` theo `run_id`. `resume(run_id)` tiếp tục
các node còn chờ của phiên workflow tương thích và giữ kết quả nhánh đã xong.
Resume là request mới có ngân sách mới, giữ execution profile đã lưu và thu hẹp
theo caller nếu caller có giới hạn chặt hơn; custom profile đã lưu không được tự
mở rộng khi khôi phục. Run đã hoàn tất không tự retry những bước `UNAVAILABLE`;
chạy phân tích mới cần run ID mới.

API công khai: `AnalysisService.resume_analysis(run_id)`. CLI tương đương:

```powershell
stock-agents --resume-run <run-uuid>
```

Mã run xuất hiện trong activity của câu trả lời và được trả cùng lỗi lấy dữ liệu
ban đầu nếu checkpoint đã lưu thành công. Hai tiến trình không được thực thi
đồng thời cùng một run.

Đây chưa phải bảo đảm exactly-once cho tác dụng phụ ngoài tiến trình. Nếu tiến
trình chết giữa I/O và commit checkpoint, operation đó có thể cần chạy lại.
UI dùng `SQLiteConversationStore` tại `.runtime/runs/conversations.sqlite3`, khôi
phục hội thoại đã lưu khi tạo lại store; thao tác xóa cũng lưu bền. Trạng thái UI
tạm thời không đồng nghĩa được khôi phục toàn bộ. Checkpoint dùng data-only
serializer, không pickle hoặc import model tùy ý; URL được lưu dưới dạng string.
SQLite checkpoint/chat chứa nội dung đầy đủ, khác telemetry metadata tối thiểu;
cần quyền truy cập, retention và backup phù hợp trước khi triển khai nhiều người
dùng. Không nạp checkpoint từ nguồn không tin cậy.

## Kiểm chứng và công việc còn lại

Lượt audit chạy lại ngày 23/10/2026: **1.023 kiểm thử đạt trong 89,74 giây**,
gồm Streamlit, HTTP loopback, lỗi specialist, deadline từng subcall và khôi phục
checkpoint sau khi tạo lại service. Provider mặc định mock, Neo4j tắt, local_hash
embedding và temp trên ổ D; các test dùng fake provider theo từng ca. Sandbox
ban đầu gặp WinError 5 ở temp; lượt có quyền thực thi phù hợp hoàn tất thành công.
Lệnh: `.venv/Scripts/python.exe -m pytest --basetemp .tmp/docs-audit-verified-20261023 --tb=short`.
Log local: `.tmp/docs-audit-verified-20261023.log`. Xem [audit tài liệu](documentation-audit-2026-10-23.md).
Không xác nhận kết nối live vnstock/Qwen từ kết quả này.

[Offline coordination benchmark](../evaluation/coordination/README.md) so sánh
single pass, bounded graph và baseline bỏ Risk chỉ trong evaluator trên cùng
input/budget. Các kiểm thử lỗi dùng fake provider/fake clock, không cần mạng.

Contract tests và benchmark tổng hợp không chứng minh chất lượng khuyến nghị
trong thị trường thật. Các việc còn lại gồm nhãn chuyên gia độc lập, dữ liệu
point-in-time đã kiểm chứng, provider schema tests, confidence calibration,
soak/load tests, phân quyền người dùng và chính sách lưu trữ.
