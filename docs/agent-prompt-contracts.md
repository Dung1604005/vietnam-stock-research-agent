# Prompt contract agent-evidence-v3-claim-anchors

Đối chiếu 23/10/2026. Goal trong AgentSpec là mô tả nhiệm vụ, không phải hàm
tối ưu được học. Các vai trò có thể dùng chung provider/model với schema khác nhau.
`AgentSpec.allowed_tools` khai báo capability, không sandbox mọi lời gọi Python;
`agent_execution.OUTPUTS/validate_update` kiểm tra quyền ghi state và sender.

| Contract | Code hiện hành |
|---|---|
| Vai trò/goal/context/tools/output/constraints | `models/agent_contracts.py`, `agents/*_agent.py` |
| Specialist review và Fundamental/Risk role prompts | `services/agent_llm.py`: SYSTEM_PROMPT, ROLE_PROMPTS, _StructuredReview |
| Technical chọn tối đa hai tool | `services/agent_llm.py`: _TechnicalIndicatorPlan, plan_technical_indicators |
| Manager typed intent/synthesis | `agents/manager_agent.py`: choose_chat_intent, summarize_response |
| RAG claim/source audit | `services/rag_service.py`, `services/claim_grounding.py` |
| Approved evidence follow-up | `services/evidence_follow_up.py` |
| Prompt version, numeric/language checks, cache/circuit | `services/llm_gateway.py` |
| Runtime output authority/correlation | `services/agent_execution.py` |

Các system prompt hiện hành vẫn nằm cạnh schema Python của từng tác vụ.
`LLMGateway` gắn version chung vào prompt và ghi model/prompt hash/status/latency;
không ghi câu hỏi, prompt, URL endpoint, token hay API key vào telemetry.

| Vai trò | Đầu vào chọn lọc | Đề xuất được phép | Quyền không được cấp |
|---|---|---|---|
| Data | ticker/horizon, provider payload và quality metadata | chuẩn hóa, xác minh provenance bằng code; không gọi LLM | tự tạo market data hoặc recommendation |
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
mỗi call ở gateway mặc định tối đa 45s, có thể cấu hình trong giới hạn 5–120s;
timeout thực tế của provider hỗ trợ timeout còn bị chặn bởi thời gian request/task
còn lại. Không có mức tối thiểu 1s khiến một call vượt phần ngân sách còn lại.

Deadline là cooperative: kiểm tra trước/sau tool, trước từng attempt, sau pacing
và backoff; `BudgetExceeded` không phải lỗi upstream để retry hoặc mở circuit.
Các adapter OpenAI/urllib dùng timeout transport đã clamp. vnstock explorer hiện
không expose per-call timeout cho mọi method; call đang chạy có thể trả về sau
deadline, khi đó kết quả muộn bị chặn và không bắt đầu outbound call tiếp theo.
Không tạo background worker để giả lập hard timeout. Finalization, báo cáo partial
và telemetry được phép hoàn tất sau khi dừng công việc có ngân sách.

Claim anchors, allowlist tool và typed message là kiểm soát runtime có phạm vi;
không biến nội dung nguồn thành chỉ dẫn đáng tin cậy và không chứng minh semantic
truth. Cold model/Colab có thể không hoàn thành; deterministic fallback hiển thị
trạng thái không khả dụng. Xem [reliability bounds](multi-agent-reliability.md).
