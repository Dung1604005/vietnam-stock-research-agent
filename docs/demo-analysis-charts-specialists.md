# Demo: biểu đồ, giải thích và LLM Fundamental/Risk

## Thay đổi

Fundamental và Risk dùng provider chung đã cấu hình (Colab Qwen nếu
`STOCK_AGENT_LLM_PROVIDER=colab_qwen`), không khởi động Ollama mới.

- Fundamental nhận metrics, kỳ/đơn vị, quan sát báo cáo, kết quả Python, nguồn
  và research chunks; nhận xét tăng trưởng, chất lượng lợi nhuận, định giá,
  dòng tiền và giới hạn ngành.
- Risk nhận kết quả Technical/Fundamental/Risk cuối cùng, nguồn, research và
  forecast; phản biện, giải thích từng nhóm rủi ro, kịch bản bất lợi có điều kiện.
- Trong full graph, hai review chạy **sau khi xử lý xung đột kết thúc**, mỗi
  role một lần trên evidence cuối. Không gọi LLM lại trong từng vòng rerun.
- Hai review có thể giúp Manager tổng hợp, nhưng không tự sửa giá, công thức,
  điểm số, veto hoặc nhãn kết luận. Không hứa làm dự báo chính xác hơn chỉ vì
  thêm lời nhận xét LLM. Failure hiển thị là failure, không gắn nhãn lời mẫu là LLM.

Đã bật profile trong `.env` hiện tại, giữ nguyên key/token:

```dotenv
STOCK_AGENT_LLM_SPECIALISTS_ENABLED=true
STOCK_AGENT_LLM_SPECIALIST_ROLES=technical,fundamental,risk
STOCK_AGENT_LLM_CALL_TIMEOUT_SECONDS=90
```

Full analysis demo mặc định cho phép 4 calls: Technical chọn tool, Fundamental
review, Risk review, Manager synthesis nếu bật. NLU repair/retry cũng dùng cùng
quota, có thể khiến synthesis/review cuối bị bỏ qua. Deadline toàn request 180s,
response review tối đa 1024 generated tokens (tương thích server Colab hiện có).
Có thể chỉnh `STOCK_AGENT_LLM_ANALYSIS_MAX_CALLS` (2–6) và
`STOCK_AGENT_ANALYSIS_DEADLINE_SECONDS` (30–300). Các con số là trần, không phải
thời gian đợi bắt buộc. Đây là nới ngân sách có chủ đích so với profile 2 calls
ở roadmap ban đầu để phục vụ demo ba specialists.

Scanning/batch vẫn `allow_llm_planning=False`, không gọi LLM cho từng mã.
Fact-mapping tùy chọn mặc định tắt để không lấy mất call dành cho review.

## Biểu đồ

Ngoài giá/nến/khối lượng/MA/RSI/MACD đã có, thêm Bollinger, drawdown, biến động
20 phiên năm hóa, lợi suất phiên, doanh thu/lợi nhuận, dòng tiền, tăng trưởng,
P/E–P/B, ROE–ROA/biên lợi nhuận. MA100/MA200 được render khi có đủ lịch sử.

Ví dụ chat:

- `Phân tích FPT`
- `Phân tích cơ bản FPT`
- `Đánh giá rủi ro FPT`
- `Vẽ dashboard FPT`
- `Vẽ RSI và MACD FPT 6 tháng`
- `Vẽ Bollinger FPT`
- `Vẽ biểu đồ sụt giảm FPT`
- `Vẽ doanh thu lợi nhuận FPT`
- `Vẽ tăng trưởng doanh thu FPT`
- `Vẽ dòng tiền FPT`
- `Vẽ ROE ROA FPT`

Nhiều chart hiển thị theo tab. Mỗi chart có nguồn, timestamp, công thức/cách đọc.
Chart tài chính giữ từng unit/frequency ở panel riêng, loại giá trị xung đột và
không dựng lịch sử từ một tỷ số hiện tại. Chuỗi tài chính hiển thị các kỳ hiện
có trong dữ liệu provider trả về, không tự tải toàn bộ lịch sử báo cáo. Thiếu
kỳ/đơn vị/chỉ tiêu thì báo không vẽ được. Warm-up thiếu được để trống, không
điền RSI trung tính để khiến chart trông đầy đủ.

## Giải thích trên app

Kết quả có “Vì sao hệ thống đưa ra nhận định này?”, bảng chỉ số doanh nghiệp
kèm cách đọc/kỳ/nguồn, bảng đóng góp điểm và “Rủi ro đến từ đâu?”. Đây là
giải thích policy và bằng chứng, không phải chứng minh nguyên nhân biến động
giá. Sau đó hiển thị riêng nhận định LLM bằng tiếng Việt và source IDs.

Khởi động lại app để áp dụng config và runtime profile mới:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Colab phải đang chạy và URL/token còn hợp lệ. Test offline chỉ kiểm tra schema,
routing, source/score safety và chart calculations; không thay thế đánh giá
chất lượng nhận xét của model thật trên báo cáo doanh nghiệp.

## Kiểm tra

- Toàn bộ suite: **468 tests passed**, gồm **27 test mới** cho các chart và
  specialist review. Test kiểm tra role call một lần sau conflict, giữ score,
  tắt LLM trong batch và fallback rõ khi endpoint lỗi.
- Streamlit AppTest render hai câu trả lời giống nhau với tổng cộng bốn tab
  chart: **0 exceptions**, không trùng ID widget.
- Runtime pytest đã cô lập khỏi app. 85 snapshot do các lần test trước tạo
  (`source=test-fixture`) được chuyển sang bản lưu có thể khôi phục tại
  `.runtime/test-artifacts/synthetic-snapshots-quarantine.sqlite3`; không xóa
  báo cáo hoặc snapshot thị trường thật.
- Không cài thêm package, không thay artifact LSTM, không commit/push key.
