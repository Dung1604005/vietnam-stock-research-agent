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

Task plan hiện là declaration có kiểm soát của công việc dự kiến. Full analysis vẫn chạy `StockAgentGraph`; technical/fundamental/risk route vẫn dùng `AnalysisService`, và chart/market/discovery/comparison vẫn là methods hiện có của `ChatService`. Không có claim rằng generic DAG executor đã tồn tại.

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

Report JSON được lưu ở `.runtime/evaluations/`. Holdout được truyền riêng qua `--holdout`; evaluator từ chối query trùng với regression sau chuẩn hóa. Thư mục [evaluation/semantic_holdout](../evaluation/semantic_holdout) chỉ có hướng dẫn, không có data gán nhãn giả.

## Safety boundary

LLM chỉ trả `SemanticQuery`; deterministic entity extraction vẫn ghi đè ticker/indicator an toàn. LLM không thể thêm route/tool tự do, tự gọi code, tăng budget hoặc bỏ dependency validator.

## Kiểm thử

- Typed dependencies và budget của full-analysis plan.
- Rejection cho forward dependency và missing input.
- Structured LLM success, safe deterministic fallback khi LLM lỗi, và parser telemetry.
- Full semantic regression dataset 120 cases.

## Gates còn mở

- Thu thập holdout độc lập từ query thật đã ẩn danh và được gán nhãn ngoài example bank khi hệ thống có traffic/tester.
- Dùng holdout để calibrate confidence theo intent/entity/clarification và đặt threshold có căn cứ.
- Thêm alerting/retention policy cho telemetry và provider usage/token/cost khi adapter hỗ trợ usage metadata.
- Chỉ xây generic task executor khi nhiều route thực sự cần scheduling/retry/dependency runtime ngoài graph hiện hữu.
- Thêm route `RESEARCH/EVIDENCE` khi Phase C có retrieval service; không tạo route giả trước đó.
