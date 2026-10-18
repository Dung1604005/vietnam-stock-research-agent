# Tự mở website và tương tác trên localhost

## Cách chạy

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address 127.0.0.1
```

Mở `http://localhost:8501` hoặc `http://127.0.0.1:8501` trên máy chạy Python. Sidebar → **Tương tác trực tiếp / Theo dõi cá nhân** có **Tự mở trang trên máy này** mặc định bật, tối đa3tab/lượt (đổi1–5 nếu cần).

Không cần bấm liên kết: sau yêu cầu mới, app gọi trình duyệt mặc định bằng `webbrowser.open`, không dùng JavaScript popup, không cần cài browser/package mới. Nếu OS không mở được, UI báo rõ. Gửi yêu cầu mở tab thành công không bảo đảm website đích đã tải đúng nội dung.

Không mở browser trên server cho khách truy cập remote/tên miền public. Lịch sử chat, rerun và đổi widget không mở lại tab. OFFLINE chặn mọi tự mở trang, kể cả phản hồi cũ có nguồn Internet.

## Câu nhập dùng ngay

| Câu nhập | Kết quả |
| --- | --- |
| `Mở trang FPT` | Trang DStock đúng mã FPT |
| `Mở giá TCB` | Trang giá/lịch sử TCB, không hứa realtime |
| `Mở biểu đồ HPG` | Biểu đồ đúng mã, không cần gọi LLM |
| `Mở VNDIRECT để mua FPT` | Cổng DBOARD và trang DStock FPT, không điền/gửi lệnh |
| `Tìm báo cáo mới nhất của FPT` | Research bình thường, rồi tự mở báo cáo thực tế đủ điều kiện |
| `Mở báo cáo FPT` | URL báo cáo đúng mã đã có trong chat; giữ kỳ/ngày và cảnh báo |
| `Phân tích FPT` | Phân tích rồi mở trang doanh nghiệp, giá và báo cáo/biểu đồ trong giới hạn tab |

Điều hướng trang/giá/biểu đồ không gọi LLM/API dữ liệu. Báo cáo mới nhất vẫn phải qua tìm kiếm/kiểm độ mới; không thay bằng tài liệu cũ hoặc đoán URL. Không tự mở broker chỉ vì kết luận MUA hoặc câu hỏi “Tôi nên mua mã nào”. Chỉ mở cổng khi có yêu cầu điều hướng; nhiều mã để mua sẽ hỏi lại thay vì chọn ngẫu nhiên.

## VNDIRECT và giới hạn giao dịch

- `https://dstock.vndirect.com.vn/suc-khoe-co-phieu/FPT` là trang nghiên cứu đúng mã, không phải phiếu lệnh.
- `https://trade.vndirect.com.vn/` là cổng DBOARD. Chưa xác minh được URL công khai bảo đảm chọn sẵn mã trên phiếu lệnh sau đăng nhập. App mở cổng và mã nghiên cứu; **bạn tự đăng nhập/chọn mã trong cổng**, không giả vờ đã chọn sẵn mã.

Không lưu tài khoản, mật khẩu, OTP/cookie; không API broker, chọn loại lệnh/giá/khối lượng hoặc xác nhận mua bán. Broker tùy chỉnh chỉ nhận URL HTTPS điều hướng; `{ticker}` chỉ hữu ích nếu bạn đã xác minh nền tảng hỗ trợ. Chặn tham số lệnh và credentials.

Nguồn: [hướng dẫn VNDIRECT](https://support.vndirect.com.vn/hc/vi/articles/4402908551705), [DStock](https://dstock.vndirect.com.vn/), [DBOARD](https://trade.vndirect.com.vn/).

## Báo cáo và an toàn URL

Liên kết lấy từ structured citations, không lấy URL tùy ý trong lời văn LLM. Hiển thị riêng ngày công bố/kỳ báo cáo; không có thì ghi chưa xác định. Không trộn báo cáo giữa các mã.

Tự mở báo cáo chỉ cho nguồn T1/T2 trong `config/research_sources.json` và URL đúng chính sách đường dẫn. Host chưa tin cậy/kết quả tìm kiếm chưa xác minh không tự mở; UI báo hạn chế và giữ evidence để review. Kiểm tra này không bảo đảm website không redirect/đổi nội dung về sau.

Chỉ HTTPS công khai; chặn localhost/IP nội bộ, token/credentials, scheme thực thi và URL lệnh. Có giới hạn tab và ledger chống mở lặp; browser lỗi không tự thử lại trên rerun, yêu cầu mới mới có thể mở lại.

## Tương tác bổ sung

- Watchlist và ghi chú riêng theo mã lưu local; bỏ theo dõi không xóa ghi chú.
- Gửi nhanh phân tích, biểu đồ, tìm báo cáo mới nhất; so sánh2–5mã trong watchlist.
- Dưới phản hồi có chọn mã, tài liệu phân trang, provenance và link mở lại dự phòng; không bắt buộc bấm để tự mở hoạt động.
- Audit mở trang: `response.metadata.browser_actions`, không thay kết luận đầu tư.

File riêng: `.runtime/interaction/workspace.json` (hoặc dưới `STOCK_AGENT_RUNTIME_DIR`), được Git ignore, không gửi LLM mặc định. Không chia sẻ app ra Internet nếu có ghi chú riêng. Xem [workspace](local-workspace.md).

Kiểm thử dùng browser spy/fixture và AppTest, không mở browser thật, đăng nhập VNDIRECT hay đặt lệnh. Bao phủ đúng mã/nguồn, URL nguy hiểm, giới hạn/dedup, OFFLINE/remote, lỗi browser và persistence.
