# Phase B — Semantic Understanding & Dynamic Routing

## Trạng thái

Nền tảng Phase B đã được triển khai mà không thay đổi financial-analysis policy hoặc thêm model/dependency. `synthetic-holdout-v1` đạt 40/40 sau khi xử lý các lỗi benchmark phát hiện; vì benchmark là dữ liệu giả do development tạo, nó là acceptance test chứ không phải independent holdout. Phase B được kết thúc ở mức synthetic acceptance theo quyết định sản phẩm hiện tại, không phải production calibration.

## Thay đổi đã có

### Bounded task plan

`chat/router.py` thay `steps` đơn thuần bằng `RoutingTask` có:

- `id`, `RouteStep` enum allowlist và dependency;
- required inputs và expected outputs;
- timeout 1–120 giây và budget units giới hạn;
- `RoutingPlanValidator` kiểm tra duplicate ID, self/forward dependency, input chưa được tạo và tổng budget trước dispatch.

`RoutingPlan.steps` còn là compatibility view cho UI/caller cũ. `ChatService` validate thêm một lần trước dispatch và trả lỗi an toàn nếu plan không hợp lệ.

Đối chiếu 23/10/2026: task plan đã được gắn với thực thi qua `ChatPlanExecution` và `PlanExecutor`, kiểm tra task được duyệt, dependency, artifact/schema thực tế và trạng thái hoàn thành. Full analysis vẫn chạy `StockAgentGraph`; selective/batch dùng service hiện có như các capability/composite có giới hạn. Đây không phải executor cho code/DAG bất kỳ do LLM tự sinh.

### Semantic telemetry

`ParsedQuery` và response metadata hiện có:

- parser source, tổng parser latency và LLM latency;
- LLM attempted, safe fallback reason (ví dụ `llm_error:RuntimeError`);
- ambiguity/clarification state;
- prompt version và example-bank version.

Parser ghi structured application log, không ghi prompt raw hay hidden reasoning. Token/cost để `unknown` vì các provider adapters hiện không trả usage metadata; không được ước lượng chi phí.

Metadata parser được lưu bền trong `.runtime/telemetry/semantic.sqlite3`: source, latency, LLM attempt/fallback, ambiguity count, clarification và version. Database **không có** cột raw query, ticker hay conversation content. Xem summary local bằng:

```powershell
.\.venv\Scripts\python.exe scripts\semantic_telemetry_report.py
```

## Evaluation và holdout

`scripts/evaluate_semantics.py` báo intent/ticker/period metrics cùng confidence calibration bins và Expected Calibration Error (ECE). Nó dùng local parser, không gọi LLM/provider dữ liệu và không ghi telemetry evaluation.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_semantics.py --show-errors --write-report
```

Report JSON được lưu ở `.runtime/evaluations/`. Holdout được truyền riêng qua `--holdout`; evaluator từ chối query trùng với regression sau chuẩn hóa. Thư mục [evaluation/semantic_holdout](../evaluation/semantic_holdout) có hướng dẫn và synthetic-v1.json được ghi rõ là synthetic smoke/acceptance, không phải query thật hoặc holdout độc lập.

## Safety boundary

LLM chỉ trả `SemanticQuery`; deterministic entity extraction vẫn ghi đè ticker/indicator an toàn. LLM không thể thêm route/tool tự do, tự gọi code, tăng budget hoặc bỏ dependency validator.

## Kiểm thử

- Typed dependencies và budget của full-analysis plan.
- Rejection cho forward dependency và missing input.
- Structured LLM success, safe deterministic fallback khi LLM lỗi, và parser telemetry.
- Full semantic regression dataset 126 cases (chạy lại 23/10/2026, intent/ticker 126/126, ECE 0.0507).

## Gates còn mở

- Thu thập holdout độc lập từ query thật đã ẩn danh và được gán nhãn ngoài example bank khi hệ thống có traffic/tester.
- Dùng holdout để calibrate confidence theo intent/entity/clarification và đặt threshold có căn cứ.
- Thêm alerting/retention policy cho telemetry và provider usage/token/cost khi adapter hỗ trợ usage metadata.
- Task executor có kiểm output và route RESEARCH đã có; scheduling tùy ý/code generation không thuộc phạm vi hiện tại.
