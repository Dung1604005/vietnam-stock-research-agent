# Chính sách báo cáo hiện tại — 23/10/2026

Sau phản hồi dữ liệu giá 2026 nhưng tài chính 2018:

- Snapshot ratio gốc PNJ (tải 23/10/2026) có các cột/kỳ đều 2018; không được đổi nhãn thành 2026.
- Fundamental ưu tiên kỳ hợp lệ mới nhất của **từng chỉ số**, kể cả khi aggregate trong cache còn là giá trị cũ hoặc thiếu. Khi chưa có kỳ mới, dùng kỳ gần nhất có thể, kể cả kỳ quá 550 ngày. Kỳ không xác định, tương lai, công bố sau thời điểm tải, đơn vị chưa xác minh (`unknown_ratio`) hoặc các giá trị mâu thuẫn cùng kỳ không được dùng; thử kỳ hợp lệ trước đó. Không sửa ngày/kỳ của dữ liệu gốc.
- Kết quả ghi `data_mode=LATEST_AVAILABLE`, `metric_periods`, `fallback_metrics` và giải thích bằng tiếng Việt khi dùng kỳ cũ. Hệ số độ tin cậy theo tuổi kỳ được dùng: đến 550 ngày = 1; đến 3 năm = 0,65; đến 5 năm = 0,45; xa hơn = 0,25. Đây là trọng số chính sách, không phải xác suất dự đoán. Chỉ số legacy không có quan sát/nguồn tài chính vẫn được xử lý theo cơ chế cũ, với nhãn `UNVERIFIED`.
- Growth/cash-conversion derived dùng cùng kiểm tra đơn vị, kỳ, basis và consolidation như trước; cho phép kỳ gần nhất đã cũ. Nếu phép tính từ statement mới hơn ratio tương ứng thì ưu tiên statement. Không lấy độ mới của chỉ số khác để xác nhận kỳ.
- Risk vẫn loại chỉ số quá 550 ngày khỏi đánh giá rủi ro hiện tại. Manager nhận riêng hạn chế thời gian của Fundamental và nêu rõ đánh giá dựa trên kỳ gần nhất; dữ liệu quá cũ không tự trở thành bằng chứng cho kết luận MUA hiện tại.
- Khi ratio thiếu/cũ, provider thử tải statement series ngay cả khi cờ tải statement bổ sung đang tắt. Đây là recovery có deadline; không bảo đảm nguồn trả báo cáo mới hoặc đủ accounting metadata.
- Audit bổ sung: vnstock VCI cục bộ `_get_report` dùng `head(4)` trước khi chuẩn hóa, và `ratio(period="year")` gắn các quý cùng năm thành một nhãn năm. Provider hiện dùng reader có kiểm tra signature, lấy lịch sử có giới hạn 512 bản ghi, giữ nguyên năm/quý và giá trị ratio để chọn kỳ gần nhất. Nếu phiên bản/nguồn không hỗ trợ reader này thì dùng public method. Không sửa thư viện trong `.venv`. Cache schema tăng lên v8 để dữ liệu đã chuẩn hóa sai không chặn lần tải mới.
- Research phân tích CURRENT luôn kiểm web nếu được phép, kể cả local có citation. Không phụ thuộc cờ LLM planning; OFFLINE không gọi mạng. Trong một run có thể dùng cache để tránh cùng một truy vấn lặp; run mới kiểm lại.
- Search dùng provider/API key hiện có, bounded 3 query/4 trang, chịu ngân sách toàn request. Kho local không bị tự xóa hoặc cấp phép ingest toàn văn.
- Sau khi gộp local/web, áp freshness rồi ưu tiên báo cáo tài chính và kỳ mới trước khi giới hạn top-k. Không được giữ top-k cũ rồi loại mất báo cáo web mới.
- Tài liệu chưa xác minh ngày công bố/kỳ không tự thành citation tài chính hiện tại; kết quả ghi rõ lý do. Tài liệu lịch sử có nhãn riêng.
- Fundamental chỉ UNKNOWN khi không còn chỉ số/chuỗi báo cáo hợp lệ có thể đánh giá, kể cả dữ liệu gần nhất. WATCH khi còn hạn chế bằng chứng không có nghĩa là doanh nghiệp yếu. Không đổi threshold BUY hoặc ép nhãn để tránh WATCH.

Code: `services/financial_freshness.py`, `agents/fundamental_agent.py`, `agents/risk_agent.py`,
`agents/research_agent.py`, `providers/market_provider.py`, `chat/response.py`.

Kiểm tra nguồn thật PNJ sau sửa adapter (23/10/2026): giá đến 23/10/2026;
240 quan sát tài chính, kỳ ratio mới nhất **30/06/2026**. Fundamental không dùng LLM:
**HOLD, 7,1/10, confidence 0,71, data_mode=CURRENT**. Đây là kết quả kiểm tra
pipeline, không phải xác nhận đã có báo cáo quý 3 hoặc đã quét mọi nguồn.
Cache mới được lưu dưới schema v8; snapshot và các kết quả hội thoại cũ giữ nguyên.

Live smoke PNJ bằng API key đã cấu hình, 2 query/3 trang/55 giây: tìm 8 URL,
trích xuất 3 nội dung, **0 citation xác minh được ngày công bố**. Không tuyên bố
đã có báo cáo mới đủ điều kiện từ kết quả đó. “Kiểm tra mới nhất” là kiểm tra theo
nguồn/kỳ trong ngân sách, không phải bảo đảm đã quét mọi website hoặc đã cải thiện
accuracy đầu tư. Cần chạy yêu cầu phân tích mới; báo cáo hội thoại cũ không được
thay đổi hồi tố. SERVICE_RUNTIME_PROFILE đã tăng để app tạo service mới.
