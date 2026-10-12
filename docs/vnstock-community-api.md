# Vnstock Community API key

Project đọc khóa từ biến môi trường chính thức `VNSTOCK_API_KEY`. Khóa thật chỉ nằm trong `.env` local trên ổ D; `.env` đã được Git ignore. Không chạy `vnai.setup_api_key(...)` vì cách đó có thể lưu thêm file vào Windows user profile trên ổ C.

## Cấu hình

Mở `.env` và dán khóa sau dấu `=`:

```dotenv
VNSTOCK_API_KEY=your_real_key
```

Không đặt dấu nháy và không gửi khóa qua chat, commit Git hoặc ảnh chụp màn hình.

Để `STOCK_AGENT_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS=` trống. Provider sẽ tự chọn:

- Không có key: `3.2` giây/lượt, phù hợp mức Guest 20 lượt/phút.
- Có key: `1.05` giây/lượt, khoảng 57 lượt/phút để chừa biên dưới mức Community 60 lượt/phút.

Kiểm tra mà không in khóa:

```powershell
.\.venv\Scripts\python.exe scripts\check_vnstock_api_key.py
```

Kết quả mong đợi có `tier=free` và `limit=60/phút`. Nếu vẫn là `guest`, kiểm tra key, dừng hẳn Streamlit rồi mở lại; loader không thay đổi biến môi trường của process đang chạy.

Khởi động lại app:

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

Nếu deploy Streamlit Community Cloud, thêm vào **App settings → Secrets**:

```toml
VNSTOCK_API_KEY = "your_real_key"
STOCK_AGENT_PROVIDER_MIN_REQUEST_INTERVAL_SECONDS = ""
```

API key tăng quota nhưng không loại bỏ cache, circuit breaker hoặc cơ chế dừng khi provider trả rate-limit.
