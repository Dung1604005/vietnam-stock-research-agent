# Demo Phase 8: chạy thật, replay rõ nguồn, không gọi dịch vụ ngoài khi offline

Đây là hướng dẫn vận hành, không phải bằng chứng model dự báo có lợi nhuận. Không có broker hoặc đặt lệnh. Chưa có số đo latency live trên máy/Colab của bạn; các trần thời gian dưới đây là **budget**, không phải thời gian cam kết.

## 1. Khởi động trên ổ D

Mở PowerShell ở thư mục repository thực tế, ví dụ `D:\Download\thayque\Agent`. Không cần activate venv và không dùng `streamlit.exe` nếu đã chuyển thư mục vì launcher có thể giữ đường dẫn cũ.

```powershell
$env:TEMP = (Join-Path (Get-Location) '.tmp')
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Máy mới: chạy `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` một lần. Script cài app/dev vào `.venv` và cache/temp trong repository trên D; không tự cài extras ML/E5, không tải model. Nếu chính Python trong `.venv` không chạy sau khi đổi đường dẫn, cần tạo lại venv tại đường dẫn mới; không sửa launcher bằng tay hoặc xóa dữ liệu dự án.

## 2. Chọn chế độ trong sidebar

| Chế độ | Hành vi thực tế | Trần profile |
| --- | --- | --- |
| STANDARD | Giữ budget hiện hành theo intent | Không tự thay `.env` |
| QUICK | Không generation/cached review LLM; có thể lấy dữ liệu thật qua provider | 0 LLM, tối đa 80 tool, 30 giây |
| DEEP | Các agent chia sẻ budget; không phải mỗi agent một quota riêng | Tối đa 4 LLM, 250 tool, 180 giây |
| OFFLINE | KB/replay local; từ chối HTTP/LLM ngoài; mã thiếu báo thiếu, không fallback Internet | 0 LLM, tối đa 250 tool, 120 giây |

Trần profile chỉ siết budget intent, không nâng intent có budget nhỏ hơn lên mức tối đa. Deadline là kiểm tra hợp tác trước tác vụ và giới hạn timeout khi transport hỗ trợ, không phải bảo đảm ngắt mọi hàm Python đang chạy. QUICK không đồng nghĩa offline hoặc không gọi Vnstock.

## 3. Chuẩn bị replay thật trước buổi trình bày

Lệnh dưới đây **có gọi dữ liệu thị trường thật** và dùng quota Vnstock; chỉ chạy khi có Internet và quyền lưu dùng nội bộ. Không tự upload/public dữ liệu.

```powershell
.\.venv\Scripts\python.exe scripts\capture_demo_bundle.py FPT VNM HPG VNINDEX `
  --output .runtime\demo\demo-2026-10-18.json `
  --rights-note "Private classroom demonstration only; no public redistribution; subject to source permission."
```

`rights-note` là khai báo của bạn, **không phải** hệ thống đã kiểm tra giấy phép thay bạn. Capture tối đa 20 mã/lần, mặc định 550 ngày lịch. VNINDEX giúp có benchmark nếu dữ liệu lấy được. Lệnh in số mã/thanh giá thành công và số lỗi; mã thất bại không được tự sinh dữ liệu thay. Nếu file đã tồn tại, chọn tên mới để giữ bản cũ.

Bundle JSON chứa full OHLCV, đơn vị gốc (UNKNOWN vẫn UNKNOWN), facts và provenance có ngày gốc; envelope có SHA-256, giới hạn 64 MiB, không pickle/giải nén/chạy code. Raw provider request parameters bị bỏ. Lưu `.runtime/demo/` trên D, không commit Git. Snapshot phân tích chỉ có last bar **không thể dùng thay bundle**.

Trong app chọn OFFLINE rồi nhập đường dẫn tuyệt đối đến JSON vừa tạo. Thử `Phân tích FPT`. Kết quả phải có nhãn `SNAPSHOT_REPLAY` và ngày nguồn gốc, không được gọi là giá hôm nay. Kết quả phân tích được tính lại từ thanh giá đã chụp; không giả rằng review LLM cũ là model vừa chạy. Nếu chọn mã không nằm trong bundle, app báo thiếu thay vì gọi Internet. Nếu checksum sai, bundle bị từ chối.

## 4. Kịch bản demo 5–10 phút

1. Mở **Tình trạng hệ thống — kiểm tra local**: readiness chỉ đọc disk/SQLite/manifest/circuit quan sát, không generate LLM hay lấy quote. `UNKNOWN`/`CONFIGURED_UNPROBED` không có nghĩa model hỏng, cũng không khẳng định model đang phục vụ.
2. QUICK: hỏi `P/E là gì?`, xem nguồn KB.
3. STANDARD/DEEP khi Colab khả dụng: `Phân tích FPT`, xem các node STARTED/SUCCEEDED/PARTIAL/FAILED phát sinh khi thực sự chạy. Không dùng phần trăm tiến độ giả; attempt mới phân biệt vòng phân tích lại.
4. Hỏi `Vẽ RSI, doanh thu và dòng tiền FPT`, xem nhiều chart, kỳ dữ liệu, đơn vị và phần thiếu. Thiếu lịch sử tài chính không được lấp bằng số giả.
5. Xem Technical/Fundamental/Risk, bảng đóng góp điểm và phần **Đối chiếu nhận định → trích đoạn → nguồn**. Không coi review legacy chỉ có source ID là claim đã được kiểm chứng đầy đủ.
6. Chọn OFFLINE với bundle thật, chạy lại một mã. Mở ngày nguồn gốc để giải thích khác biệt live/replay. Nếu demo conflict replay thì nói rõ đó là tình huống lịch sử, không giả live conflict.
7. Tải Markdown/HTML/audit JSON. Kiểm tra nội dung trước khi chia sẻ.

Các câu hỏi chỉ là gợi ý: dữ liệu có thể thiếu và tín hiệu không bị ép ra MUA. Không đảm bảo mọi nguồn/nguồn tài chính/Colab đều khả dụng trong ngày trình bày.

## 5. Report và quyền riêng tư

Ba định dạng dùng chung sanitizer/allowlist: có nguồn, timestamps, chart records/spec, các review/claim audit và policy khi response cung cấp. Không dump toàn bộ AgentState/metadata, raw prompt, request parameters, credentials hoặc local paths. Query/fragment/userinfo trong URL bị bỏ; liên kết ký bằng query có thể cần truy cập trang nguồn để mở lại.

Phân tích portfolio bị lược cả nội dung/ticker/bảng/chart mặc định; không chỉ xóa weights. Python API có opt-in rõ `build_audit(response, include_private_portfolio=True)`; UI không bật chia sẻ riêng tư mặc định. HTML được escape, không nhúng chart tương tác. Sanitizer là biện pháp kỹ thuật hỗ trợ, vẫn cần đọc lại báo cáo trước khi chia sẻ.

## 6. Kiểm tra trước buổi demo

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_demo_runtime_upgrade.py
.\.venv\Scripts\python.exe scripts\evaluate_upgrade.py --strict
```

Test fixture/regression kiểm tra engineering, không chứng minh độ chính xác đầu tư. Health không load TensorFlow/GPU và không hash model lớn mỗi rerun: `PRESENT_UNLOADED` chỉ xác nhận manifest/file khai báo hiện diện, `model_hashes_verified=false`; inference thực tế phải tự kiểm tra hash/schema. Database sai bảng báo `SCHEMA_MISMATCH`, bundle hỏng bị từ chối.

| Nội dung đo | Đã có bằng chứng | Chưa có bằng chứng |
| --- | --- | --- |
| Profile, replay checksum, nguồn/ngày/đơn vị, export/privacy, progress | Test engineering offline | Tốc độ máy người dùng |
| Local health | Test 20 lần render: 0 kết nối ngoài | Uptime provider/Colab |
| Fine-tuned LLM/LSTM V3 | Có pipeline/notebook và gate riêng | Chưa train/A-B/đạt gate chỉ vì app chạy |

Giữ V2 và base model làm rollback cho đến khi artifacts mới vượt gate. Không sửa `.env` hoặc quảng bá adapter được duyệt chỉ vì training loss giảm.
