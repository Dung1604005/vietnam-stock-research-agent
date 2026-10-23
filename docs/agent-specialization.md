# Bài thuyết trình: chuyên môn hóa vai trò agent và code tương ứng

Đối chiếu code 23/10/2026. Bài nói khoảng 4–6 phút tùy tốc độ, dùng cùng
[báo cáo hiện trạng](project-report.md) để tránh lấy trạng thái từ kế hoạch cũ.

## Bài nói

“Trong dự án nghiên cứu cổ phiếu, nhóm em tổ chức sáu vai trò: Data, Research,
Technical, Fundamental, Risk và Manager. Specialization nghĩa là mỗi vai trò
có mục tiêu, dữ liệu đầu vào, công cụ, đầu ra và giới hạn quyền quyết định riêng.
Trong code, hợp đồng đó là AgentSpec; chúng em không chỉ đặt tên khác nhau cho
cùng một prompt.

Data chịu trách nhiệm lấy và chuẩn hóa dữ liệu thị trường có nguồn, thời điểm và
kiểm tra chất lượng. Agent này không gọi LLM và không đưa ra khuyến nghị. Research
tìm tài liệu có trích dẫn, kiểm tra kỳ và độ mới. Nó phân biệt tài liệu đã đọc
với kết quả tìm kiếm mới chỉ có đường link, không biến một bài viết thành tín hiệu mua.

Technical và Fundamental nhận hai nhiệm vụ riêng. Technical đánh giá xu hướng
bằng chỉ báo giá, khối lượng và dự báo đủ điều kiện. LLM có thể chọn tối đa hai
chỉ báo bổ sung trong allowlist; Python tính số và chấm điểm. Fundamental đánh giá
tăng trưởng, chất lượng tài chính và định giá theo dữ liệu/kỳ/ngành. Thiếu số thì
báo thiếu, dữ liệu yếu thì giảm độ tin cậy. Cách tách này giữ riêng hai câu hỏi:
giá đang diễn biến thế nào và nền tảng doanh nghiệp ra sao.

Trong full graph, hai nhánh được chạy song song rồi join trước Risk. Risk xem
cả hai kết quả, đo biến động, mức sụt giảm, thanh khoản và yếu tố doanh nghiệp hoặc
danh mục khi có dữ liệu. Risk phân biệt rủi ro đo được với bất định do thiếu dữ liệu.
Nó có quyền phát veto theo rule; Manager không thể cho ra BUY khi veto còn hiệu lực.

Manager lập kế hoạch, phân công và xem các kết quả có mâu thuẫn hay không. Nó có
thể yêu cầu mở rộng Technical, bổ sung Research/Fundamental hoặc làm mới Data.
Các vòng bị giới hạn bởi lịch sử hành động, thay đổi bằng chứng và ngân sách,
không được tranh luận vô hạn đến khi ra một kết luận mong muốn. Quyết định cuối
vẫn do Python policy, còn LLM chỉ hỗ trợ routing, đọc bằng chứng và diễn đạt.

So với gộp vào một agent, cách tổ chức này giúp truy vết vấn đề theo từng khâu,
kiểm thử và thay thế từng module, đồng thời áp dụng giới hạn quyền rõ ràng. Ví dụ
Technical chỉ được ghi technical_result, không được tự sửa risk_result hoặc
final_decision; runtime kiểm tra điều đó. Tách Risk cũng làm bước phản biện trở
thành yêu cầu của workflow trước khi kết luận.

Đổi lại, nhiều vai trò cần thêm code điều phối, truyền state, xử lý lỗi và chi
phí gọi công cụ. Chúng không tự tạo ra nguồn thông tin độc lập nếu cùng dùng một
model và dữ liệu. Vì vậy nhóm em không khẳng định nhiều agent mặc nhiên chính
xác hơn. Benchmark hiện có chỉ so graph một lượt với graph có tái phân tích và
baseline bỏ Risk, trên bốn tình huống tổng hợp; nó chưa so sánh công bằng với
một LLM agent duy nhất. Giá trị đã thể hiện trong code là kiểm soát trách nhiệm
và quá trình ra quyết định có thể kiểm tra.”

## Thứ tự mở code khi trình bày

| Ý cần chứng minh | Code / symbol |
|---|---|
| Role, goal, input/tool/output/constraints | [AgentSpec](../src/stock_multi_agent/models/agent_contracts.py); `spec` trong sáu file agents |
| Data không dùng LLM | [DataAgent.spec](../src/stock_multi_agent/agents/data_agent.py): can_use_llm=False; BaseAgent.llm_review kiểm tra cờ |
| Research dùng citation, không signal | [ResearchAgent.run](../src/stock_multi_agent/agents/research_agent.py) |
| Technical dùng tool Python, optional LLM planner | [TechnicalAgent.run/_apply_lstm_blend](../src/stock_multi_agent/agents/technical_agent.py), [schema _TechnicalIndicatorPlan](../src/stock_multi_agent/services/agent_llm.py) |
| Fundamental tự chấm điểm và quality weighting | [FundamentalAgent.run](../src/stock_multi_agent/agents/fundamental_agent.py) |
| Risk và veto | [RiskAgent.run](../src/stock_multi_agent/agents/risk_agent.py): `has_trading_suspension or issuer_score >= 8.5` |
| Song song và join | [StockAgentGraph._build](../src/stock_multi_agent/graph/stock_agent_graph.py): dispatch_specialists → technical/fundamental; join → dispatch_risk |
| Giới hạn quyền ghi | [OUTPUTS/validate_update](../src/stock_multi_agent/services/agent_execution.py) |
| Ba hướng re-analysis và chặn vòng | [Manager.review/route_after_review](../src/stock_multi_agent/agents/manager_agent.py) |
| Policy kết luận | [Manager.finalize](../src/stock_multi_agent/agents/manager_agent.py) |
| Phân biệt kiểm chứng contract với accuracy đầu tư | [AgentEvaluationService](../src/stock_multi_agent/services/agent_evaluation.py), [benchmark](../evaluation/coordination/README.md) |

Ví dụ giả định để giải thích: Technical BUY, Fundamental HOLD thì Manager nhận
thấy bất đồng; nếu dữ liệu còn có thể bổ sung thì chọn một nhánh kiểm tra lại.
Nếu có Risk veto, policy hiện tại kết luận AVOID; không dùng ví dụ đó để nói hệ
thống sẽ luôn chạy lại Technical hoặc kết luận WATCH/HOLD. Tín hiệu của FPT thật
không được ép cố định theo ví dụ.

## Câu hỏi có thể gặp

- **Có sáu model không?** Không. Sáu vai trò có thể chia sẻ một provider/model với schema/prompt riêng; Data không gọi LLM.
- **Có phải hoàn toàn tự chủ không?** Không. Đây là bounded supervisor/shared state; route/action templates và policy vẫn bằng code. Autonomous goal/worker mới là kế hoạch.
- **Goal có phải hàm tối ưu đã học không?** Không. Chuỗi goal mô tả nhiệm vụ; hành vi được cụ thể hóa bằng code, contract và policy.
- **Tool allowlist có cô lập agent không?** Có allowlist cho lựa chọn LLM và kiểm quyền output; AgentSpec không phải sandbox cho mã Python trong cùng process.
- **Một agent có thể làm được không?** Có thể thiết kế một agent dùng cùng tools/guards. Muốn kết luận chất lượng hơn cần baseline cùng data, tools, budget và nhãn độc lập; benchmark hiện tại chưa có baseline đó.
