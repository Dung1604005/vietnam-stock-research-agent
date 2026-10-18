# Chọn LSTM V2 / V4 trên UI

Trong sidebar, chọn **Mô hình dự báo LSTM → V2 hoặc V4 · thử nghiệm**, rồi gửi yêu cầu mới. Không cần sửa `.env` hoặc khởi động lại app khi chuyển model. Hội thoại cũ được giữ nguyên; kết quả cũ không bị đổi model hồi tố. Service và cache screening được tạo lại khi đổi lựa chọn để không trả điểm của model trước.

Khi dự báo hợp lệ và Technical đủ dữ liệu:

```text
Điểm Technical = 0,80 × điểm chỉ báo + 0,20 × điểm tín hiệu LSTM
Điểm tín hiệu LSTM = 5 + 5 × clip(lợi nhuận dự báo / 0,03; -1; 1)
```

20% là trọng số phối hợp, không phải độ chính xác hay tỷ lệ lợi nhuận. Không cộng V2 và V4 cùng lúc; chỉ dùng model đang chọn. Fundamental, Risk và chính sách Manager vẫn hoạt động độc lập.

V4 là **experimental blend do người dùng yêu cầu**, không phải model đã qua kiểm chứng thực tế. Giữ nguyên `policy_eligibility=false`, `validation_status=PENDING_FORWARD_VALIDATION` và manifest gốc. Cờ riêng `experimental_blend_enabled=true` cho phép đóng góp 20% ở chế độ này, không giả mạo kết quả gate hoặc xác nhận giá điều chỉnh. Dự báo/diagnostics ghi phiên bản model và trạng thái thử nghiệm.

## Artifact

- V2: tiếp tục dùng artifact hiện hành trong `notebooks/trainResult`, hoặc `STOCK_AGENT_LSTM_ARTIFACT_DIR`.
- V4: tự tìm đúng **một** `lstm-v4-full-*-results.zip` trong `.runtime/ml/v4/`. Nếu có nhiều ZIP, cấu hình `STOCK_AGENT_LSTM_V4_ARCHIVE` bằng đường dẫn tới ZIP bạn chọn rồi restart app; không tự chọn bản mới nhất.
- Không cần giải nén thủ công. Adapter kiểm checksum toàn bộ file được manifest pin và identity, dùng feature/scaler/volatility/calibration đúng V4; chỉ nạp `model.keras` với `safe_mode=True`, không chạy code Python nằm trong ZIP.
- Không tải model qua Internet, không train lại. TensorFlow dùng môi trường D hiện có. Chỉ tạo file model tạm bên cạnh ZIP trên D để Keras đọc; file tạm được dọn sau khi nạp.

Nếu tắt `STOCK_AGENT_LSTM_ENABLED`, thiếu TensorFlow, thiếu/lỗi artifact, thiếu cửa sổ dữ liệu liên tục hoặc suy luận lỗi: fallback được ghi rõ, **không cho baseline hưởng trọng số 20%**, không âm thầm thay V4 bằng V2.

Sàng lọc sơ bộ nhiều mã vẫn chỉ dùng chỉ báo để giữ tốc độ; model đang chọn được dùng khi phân tích sâu các ứng viên và tính điểm Technical cuối. OFFLINE replay vẫn giữ chính sách không chạy dự báo được cấu hình của live service.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Kiểm tra kết quả mới: phiên bản trong phần dự báo, `technical_result.diagnostics.lstm_model_version`, `lstm_weight=0.2`, `indicator_weight=0.8`. Với V4, có thêm `lstm_experimental_blend=true`. Nếu không áp dụng được, không tuyên bố 20% đã được tính.
