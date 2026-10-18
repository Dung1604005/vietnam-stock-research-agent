# Prompt contract agent-evidence-v2

Các system prompt hiện hành vẫn nằm cạnh schema Python của từng tác vụ.
`LLMGateway` gắn version chung vào prompt và ghi model/prompt hash/status/latency;
không ghi câu hỏi, prompt, URL endpoint, token hay API key vào telemetry.

| Vai trò | Đầu vào chọn lọc | Đề xuất được phép | Quyền không được cấp |
|---|---|---|---|
| Manager | intent/entities hoặc bảng kết quả + nguồn | chọn typed intent, tổng hợp | thay ticker/count/score/decision |
| Research | query hoặc source chunks | chia truy vấn, claim/citation, nêu mâu thuẫn | tự coi trang tìm thấy là dữ kiện đã xác minh |
| Technical | chỉ báo Python + forecast + nguồn | chọn tối đa hai tool, nhận xét regime | tự tính RSI/MACD/giá |
| Fundamental | metrics + evidence + kỳ | drivers, phản biện/missing facts | điền số thiếu, thay score |
| Risk | kết quả specialists + evidence | kịch bản có điều kiện, critic | hủy veto hoặc thay risk score |

Đầu ra luôn qua Pydantic; phần văn bản tiếng Việt được kiểm tra ký tự/ngôn ngữ,
số xuất hiện trong evidence pack và allowlist citation. Đây là kiểm tra cơ học,
không chứng minh quan hệ ngữ nghĩa của mọi con số. Claim RAG còn có audit riêng
theo đúng cited chunk. Triple kinh tế do LLM đề xuất cần review, không auto-write.

Budget chat: simple 1 call, analysis demo 4 (profile cũ 2), deep research 3; từng role bật theo config
và có thể bị bỏ qua khi hết budget. Không tăng số call chỉ để mọi agent đều gọi.
Cache 128 entries/120s trong RAM; circuit mở sau hai lỗi, cooldown 30s. Timeout
mỗi transport mặc định 45s, profile demo 90s và không vượt deadline còn lại. Cold model/Colab có thể
không hoàn thành; deterministic fallback phải hiển thị rõ.
