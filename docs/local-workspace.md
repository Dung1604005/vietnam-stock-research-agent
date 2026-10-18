# Workspace tương tác trên localhost

`LocalWorkspace` lưu danh sách mã theo dõi, ghi chú cá nhân và cấu hình liên kết website công ty chứng khoán. Không gọi mạng, không lưu danh mục tài sản thực, mật khẩu, API key hay lệnh giao dịch.

Đường dẫn mặc định là `.runtime/interaction/workspace.json` trong repository trên ổ D. Nếu có `STOCK_AGENT_RUNTIME_DIR`, file nằm trong thư mục `interaction` bên dưới runtime đó. Đây là lưu trữ cá nhân trên máy, không phải cơ chế phân quyền nhiều người dùng: không chia sẻ app/ổ đĩa nếu ghi chú chứa nội dung riêng tư.

API:

```python
from stock_multi_agent.services.local_workspace import LocalWorkspace, WorkspaceError

workspace = LocalWorkspace()
workspace.add_ticker("FPT")
workspace.save_note("FPT", "Cần xem báo cáo quý mới nhất.")
print(workspace.list_tickers())
print(workspace.get_note("FPT"))
```

`remove_ticker()` bỏ mã khỏi danh sách nhưng giữ ghi chú. `save_note(ticker, "")` xóa riêng ghi chú. Mã được chuẩn hóa chữ hoa; ghi chú tối đa 10.000 ký tự, tối đa 1.000 mã và 1.000 ghi chú. Chỉ hiển thị ghi chú bằng thành phần văn bản, không render HTML không an toàn.

`save_preferences()` thay thế toàn bộ cấu hình, chỉ nhận `broker_name` và `broker_url_template`. Template phải là HTTPS, không chứa thông tin đăng nhập, chỉ hỗ trợ placeholder `{ticker}`. Cấu hình này chỉ dùng để mở website; không bảo đảm website có deep link đặt đúng mã và không tự điền hay gửi lệnh mua/bán.

File ghi theo cách atomic, có khóa thread và khóa process qua file `.lock` cạnh JSON. Không xóa file khóa khi app đang chạy. JSON hỏng hoặc schema không tương thích tạo `WorkspaceError` rõ ràng, không tự ghi đè. Khi gặp lỗi, dừng app, sao lưu file, sửa hoặc đổi tên JSON lỗi rồi mở app lại. Không có đồng bộ đám mây hay cập nhật giá nền trong dịch vụ này.
