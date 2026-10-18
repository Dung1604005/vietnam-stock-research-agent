# Kết nối các agent với Qwen3

## Đã có gì và chưa có gì?

Notebook mới: `notebooks/serve_qwen3_4b_colab.ipynb`, code đồng bộ `notebooks/serve_qwen3_4b_colab.py`. Đây là **inference Qwen3 gốc**, không phải notebook train và không tự dùng LoRA. Model/revision: `Qwen/Qwen3-4B-Instruct-2507` / `cdbee75f17c01a7cc42f958dc650907174af0554`.

App đã có client dùng chung cho Manager, Technical, Fundamental, Risk và các tác vụ RAG khi được định tuyến. Mỗi tác vụ gửi system prompt, evidence và JSON schema riêng; Python vẫn tính chỉ báo/điểm, thực thi tool và kiểm tra bằng chứng. Không phải mọi agent đều cần gọi LLM mỗi lượt; Data vẫn là service lấy dữ liệu, scan sơ bộ không gọi LLM từng mã.

Adapter đã tải tại `.runtime/llm-training/qwen3_lora_pilot_cuda13_v1-results.zip` là full training 2 epoch, 48 bước, dữ liệu `SYNTHETIC_DEMO_ONLY`. Tính lại từ predictions: base đúng5/48 và candidate47/48; cả48 output hợp schema. Test không có `STOCK_RECOMMENDATION`. Adapter/config và3split có checksum khớp. Đây không phải đánh giá độc lập trên câu hỏi thực tế. Adapter trả `intent/tickers/count`, Manager yêu cầu `intent/confidence`: không tự áp adapter NLU cho tất cả các agent, không sửa `PENDING` thành `APPROVED`.

Lượt triển khai này không tải/train Qwen trên laptop, không cài package lên C hoặc vào venv local, không công bố endpoint thật, không gửi prompt hay token tới URL cũ để thử. Kiểm thử server dùng generator fixture, chưa xác minh tốc độ/chất lượng GPU Colab của notebook serving mới.

## 1. Chạy inference trên Colab

1. Colab → File → Upload notebook → chọn `notebooks/serve_qwen3_4b_colab.ipynb`.
2. Runtime → Change runtime type → T4 GPU. Không chạy chung session còn giữ model training. Restart session nếu cần giải phóng GPU.
3. Chạy cell cài thư viện. Dùng pin `transformers4.55.4`, `accelerate1.10.1`, `bitsandbytes0.48.2`, `jsonschema4.25.1`. Giữ PyTorch/CUDA của Colab; native NF4 được thử trước khi tải model. Nếu cell báo kernel giữ thư viện cũ: Runtime → Restart session rồi chạy từ cell định nghĩa code, không chạy cell train.
4. Chạy cell định nghĩa code, cell nạp model và cell smoke JSON. Phải thấy đúng model, `BASE_ONLY`, rồi output `status=OK`, `language=vi`. Lần đầu cần Internet tải model vào máy ảo Colab, không phải laptop; không cần upload ZIP fine-tune cho chế độ này.
5. Chạy cell server. Server bind `127.0.0.1:8000` bên trong Colab, bắt buộc Bearer token. Laptop không thể gọi trực tiếp localhost của Colab.

## 2. Điều kiện để app bên ngoài gọi vào

**Colab Free không phải backend hosting ổn định**. FAQ hạn chế bỏ qua notebook để tương tác chủ yếu qua web UI và có thể ngắt phiên. Không dùng tunnel/keepalive để lách giới hạn. Phần công bố endpoint mặc định **tắt**; chỉ dùng khi chính sách môi trường/runtime của bạn cho phép. Nếu chỉ có Colab Free, có thể thử inference trực tiếp notebook; không hứa app bên ngoài được hỗ trợ như dịch vụ luôn chạy.

Khi có môi trường cho phép, đọc chính sách rồi đổi `PUBLISH_ENDPOINT=True`, `HOST_POLICY_REVIEWED=True` và chạy cell tunnel. Cell tải `cloudflared` từ release chính thức, mở HTTPS Quick Tunnel tạm thời rồi in URL/token/model. Không cần API LLM trả phí trong code; GPU/quota/hosting tùy môi trường của bạn. Không tự mua gói dịch vụ.

Giữ notebook hoạt động cho demo, không chia sẻ token hoặc lưu output token vào Git. Tunnel chỉ dành thử nghiệm, không có SLA. Đóng phiên xong token/URL có thể không còn dùng được.

## 3. Cấu hình app trên laptop

`.env` đã đổi tên model từ Qwen2.5 sang Qwen3; URL/token hiện có được giữ nguyên vì không biết giá trị mới. Bạn **phải thay cả URL và token** bằng giá trị phiên notebook mới. Không dùng URL server Qwen2.5 cũ rồi chỉ đổi tên model.

```dotenv
STOCK_AGENT_LLM_PROVIDER=colab_qwen
COLAB_QWEN_MODEL=Qwen/Qwen3-4B-Instruct-2507
COLAB_QWEN_MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
COLAB_QWEN_TOKENIZER_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
COLAB_QWEN_BASE_URL=https://URL-MOI.trycloudflare.com
COLAB_QWEN_API_TOKEN=TOKEN-MOI-TU-NOTEBOOK
COLAB_QWEN_TIMEOUT_SECONDS=120
STOCK_AGENT_LLM_MANAGER_ROUTING_ENABLED=true
STOCK_AGENT_LLM_MANAGER_SYNTHESIS_ENABLED=true
STOCK_AGENT_LLM_SPECIALISTS_ENABLED=true
STOCK_AGENT_LLM_SPECIALIST_ROLES=technical,fundamental,risk
STOCK_AGENT_LLM_ADAPTER_MANIFEST=
```

Giữ adapter manifest trống cho Qwen3 gốc. Không điền manifest synthetic vào registry đã được phê duyệt.

Từ thư mục dự án trên D:

```powershell
.\.venv\Scripts\python.exe scripts\check_colab_qwen.py
```

Script kiểm GET `/health` có xác thực và đối chiếu model/revision, rồi gọi một prompt JSON thật. Đã sửa lỗi script trước đây coi gateway wrapper là sai provider. Không in URL/token. Nếu báo khác model, cần thay URL hoặc dùng đúng notebook Qwen3, không sửa identity giả.

**Dừng Streamlit cũ bằng Ctrl+C**, mở lại:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Chọn STANDARD/DEEP trong UI. QUICK/OFFLINE cố ý không gọi LLM. Loader không ghi đè process secrets nên reload trang không đủ để nhận mọi thay đổi `.env`.

## 4. Biết agent có thật sự gọi model không

Thử “Phân tích FPT” hoặc “Tôi nên mua cổ phiếu nào hôm nay?”. Xem metadata:

- `manager_routing.llm_used` và `fallback_reason`: model đã được gọi và kết quả có được chấp nhận hay không.
- `llm_specialist_reviews`: Technical dùng indicator planner; Fundamental/Risk đánh giá evidence theo quyền và budget. Grounding có thể từ chối nhận định không có bằng chứng.
- `manager_synthesis`: xem trạng thái cùng lý do nếu không tổng hợp được.
- Runtime health: model phải là Qwen3; một health check tốt không bảo đảm schema của mọi tác vụ đều đạt.

Server chỉ chạy một generation mỗi lúc; bận trả429, output sai schema trả422, không tự sinh kết quả giả hoặc retry vô hạn. Giới hạn input6144token/output1024token; chọn evidence ngắn hơn nếu gặp413. Timeout phía app120giây; thời gian generation server90giây là soft bound của Transformers, không phải cam kết latency.

Dừng: đổi `STOP_SERVER=True` ở cell cuối rồi chạy; restart session để giải phóng GPU. Không có cơ chế keepalive/né quota.

## Nguồn chính thức

- [Qwen3 model card và non-thinking inference](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- [bitsandbytes0.48.2: CUDA13 trên Linux/T4](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation).
- [Colab FAQ và giới hạn sử dụng](https://research.google.com/colaboratory/faq.html).
- [Cloudflare Quick Tunnel: thử nghiệm, không có SLA](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
