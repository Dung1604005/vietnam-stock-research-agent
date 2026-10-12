# Qwen2.5-3B-Instruct trên Google Colab

Project dùng một Qwen endpoint do chính notebook Colab tạo ra. Không cần Ollama local và không cần API LLM trả phí. Colab chỉ cung cấp inference; Python trong project vẫn tính chỉ báo, điểm số, LSTM forecast, risk veto và quyết định cuối cùng.

## Thiết lập

1. Mở `notebooks/serve_qwen25_3b_colab.ipynb` bằng Google Colab.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chọn **Runtime → Run all** và chờ model tải xong.
4. Cell cuối in hai dòng `COLAB_QWEN_BASE_URL=...` và `COLAB_QWEN_API_TOKEN=...`.
5. Dán hai giá trị đó vào file `.env` local. Không commit `.env` hoặc token lên Git.
6. Kiểm tra endpoint mà không in token:

```powershell
.\.venv\Scripts\python.exe scripts\check_colab_qwen.py
```

7. Khởi động lại Streamlit để cache service được tạo lại:

```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```

Giữ tab Colab và runtime hoạt động trong lúc dùng app. URL `trycloudflare.com` và token là tạm thời; sau mỗi lần Colab restart phải cập nhật lại hai giá trị trong `.env`.

Nếu app đang chạy trên Streamlit Community Cloud, mở **App settings → Secrets** và đặt các biến tương đương trong TOML thay vì commit `.env`:

```toml
STOCK_AGENT_LLM_PROVIDER = "colab_qwen"
STOCK_AGENT_LLM_MANAGER_ROUTING_ENABLED = "true"
STOCK_AGENT_LLM_MANAGER_SYNTHESIS_ENABLED = "true"
STOCK_AGENT_LLM_SPECIALISTS_ENABLED = "true"
STOCK_AGENT_LLM_SPECIALIST_ROLES = "technical"
COLAB_QWEN_BASE_URL = "https://...trycloudflare.com"
COLAB_QWEN_API_TOKEN = "..."
COLAB_QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
COLAB_QWEN_TIMEOUT_SECONDS = "120"
```

Mỗi lần Colab tạo URL/token mới, cập nhật Secrets rồi reboot app.

## Cấu hình đề xuất

```dotenv
STOCK_AGENT_LLM_PROVIDER=colab_qwen
STOCK_AGENT_LLM_SEMANTIC_ENABLED=false
STOCK_AGENT_LLM_MANAGER_ROUTING_ENABLED=true
STOCK_AGENT_LLM_MANAGER_SYNTHESIS_ENABLED=true
STOCK_AGENT_LLM_ANALYST_ENABLED=false
STOCK_AGENT_LLM_SPECIALISTS_ENABLED=true
STOCK_AGENT_LLM_SPECIALIST_ROLES=technical
COLAB_QWEN_BASE_URL=https://...trycloudflare.com
COLAB_QWEN_API_TOKEN=...
COLAB_QWEN_MODEL=Qwen/Qwen2.5-3B-Instruct
COLAB_QWEN_TIMEOUT_SECONDS=120
```

Với profile này:

- Manager gọi Qwen để chọn một `QueryIntent` có kiểu dữ liệu, sau đó deterministic router kiểm tra quyền và lập task plan.
- Technical gọi Qwen để chọn tối đa hai chỉ báo bổ sung trong allowlist; Python tính giá trị và chấm điểm.
- Sau khi các agent hoàn tất, Manager gọi Qwen đúng một lần để viết 4–6 câu tổng hợp bám theo kết quả đã tính.
- Vòng lọc nhanh 100 mã không gọi Qwen cho từng mã; chỉ các ứng viên vào phân tích sâu mới có thể dùng Technical LLM.
- Khi endpoint timeout, tắt hoặc trả sai schema, ứng dụng giữ kết quả deterministic và ghi lý do fallback trong metadata.

## Ranh giới dữ liệu

Các request gửi đến Colab có thể gồm câu hỏi người dùng, ticker, horizon, chỉ báo đã tính, kết quả agent, cảnh báo và source ID. Phần tổng hợp chỉ gửi tối đa hai bảng nhỏ cùng phần response đã cắt giới hạn. Project không gửi file `.env`, token, mật khẩu Neo4j, toàn bộ raw cache hoặc snapshot provider.

Endpoint yêu cầu Bearer token và notebook không log request payload. Tuy vậy, tunnel là một kết nối Internet tạm thời; chỉ dùng token được notebook sinh ra và dừng Colab khi không dùng nữa.

## Kiểm tra

Sau khi chạy một câu hỏi, mở metadata trong UI:

- `manager_routing.llm_used=true` cho biết Manager đã gọi Qwen để hiểu intent.
- `manager_synthesis.status=GROUNDED` cho biết phần **Tổng hợp từ Manager LLM** đã được tạo.
- `llm_specialist_reviews` có `model_role=indicator_planner` cho biết Technical đã dùng Qwen chọn công cụ.

Nếu thấy `UNAVAILABLE`, kiểm tra Colab còn chạy, URL/token trong `.env` có đúng và tunnel chưa đổi URL.
