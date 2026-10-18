# Lọc cổ phiếu bằng điều kiện số

Nhập trực tiếp trong chat, ví dụ:

- `Tìm cho tôi 10 cổ phiếu có RSI > 30`
- `Lọc 10 mã RSI >= 30 và RSI < 70`
- `Tìm 10 mã RSI từ 30 đến 70`
- `Tìm 5 mã P/E < 15 và ROE >= 20`

Hỗ trợ RSI(14), P/E, P/B, ROE và các phép `<`, `>`, `<=`, `>=`, `=`; có thể dùng “trên”, “dưới”, “lớn hơn”, “không thấp hơn”, “tối đa”. Dấu phẩy thập phân được chấp nhận. Các điều kiện kết hợp bằng **và**; **hoặc**, chu kỳ RSI khác 14 và khoảng mâu thuẫn cần làm rõ, không tự đổi thành điều kiện khác. ROE nhập theo phần trăm: `ROE >= 20` nghĩa là từ 20%.

## Luồng thực thi

1. Parser đọc số lượng, chỉ báo, toán tử và ngưỡng từ câu gốc, trước khi chuẩn hóa mất ký hiệu toán học.
2. Router chọn Screening; truy vấn số rõ ràng không cần LLM suy đoán toán tử hoặc gọi thêm các chuyên gia sau khi lọc.
3. Screening lấy dữ liệu trong universe cấu hình, tối đa 100 mã. RSI dùng lại snapshot giá của lần đánh giá kỹ thuật; không tải giá lại chỉ để tính RSI.
4. Python tính chỉ báo và so sánh trên giá trị chưa làm tròn. Bảng kết quả hiển thị chỉ báo thực tế, ngày dữ liệu và thông tin nguồn đi cùng phản hồi.
5. Nếu nhiều mã cùng đạt, giữ số lượng yêu cầu theo thứ tự điểm kỹ thuật. Nếu không đủ, trả số thực tế và lý do, không tạo mã/dữ liệu để bù.

Truy vấn trung lập như `RSI > 30` không tự thêm điều kiện BUY/HOLD hoặc điểm kỹ thuật tối thiểu. Mã có tín hiệu SELL vẫn có thể đạt điều kiện số; **đạt bộ lọc không phải khuyến nghị mua**. Nếu câu có yêu cầu mua/bán, chính sách tín hiệu tương ứng vẫn áp dụng.

Chất lượng, độ mới và khả năng so sánh ngày dữ liệu vẫn được kiểm tra. Đây không phải quét toàn thị trường hay nguồn giá realtime bảo đảm. Lọc tài chính/sàn còn có ngân sách enrichment giới hạn (tối đa `min(100, max(20, số_lượng * 3))` ứng viên qua bước đầu); cảnh báo công khai giới hạn này. RSI-only không dùng giới hạn enrichment đó.

## Công thức RSI đang dùng

Toolkit hiện có lấy trung bình đơn giản các mức tăng/giảm trong 14 biến động giá đóng cửa gần nhất (cần ít nhất 15 giá), không phải Wilder smoothing. Vì vậy giá trị có thể khác nền tảng dùng phương pháp làm mượt khác. Không thay đổi công thức cũ trong lần nâng cấp này.

## Vai trò KB/RAG

- Câu `RSI > 30 nghĩa là gì?` vẫn đi nhánh Knowledge để giải thích lý thuyết, không quét cổ phiếu.
- RAG phục vụ hỏi báo cáo, tin tức và bằng chứng doanh nghiệp trong các yêu cầu nghiên cứu hiện có; không thay thế dữ liệu giá mới hoặc sinh giá trị RSI.
- Không bổ sung lượt tìm web/LLM cho từng mã chỉ để kiểm tra một ngưỡng số. Chưa tự động gắn một đoạn KB/RAG vào mọi kết quả lọc.

## Trình duyệt

Sàng lọc không tự mở website ở cấu hình mặc định. Muốn mở, gửi yêu cầu mở trang rồi xác nhận, hoặc tự tick cho phép ở sidebar. Xem [tương tác local](local-interactions.md).
