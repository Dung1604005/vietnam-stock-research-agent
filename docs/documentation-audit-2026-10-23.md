# Audit và cập nhật báo cáo — 23/10/2026

## Phạm vi và cách kiểm tra

Đối chiếu các mô tả kiến trúc/phân vai/luồng/runtime với working tree thực tế;
kiểm kê toàn bộ Markdown trong docs và README gốc, phân loại hướng dẫn hiện hành,
báo cáo lịch sử, kế hoạch và review model. Sửa các khẳng định đã mâu thuẫn với code,
thêm nguồn tổng hợp và ghi rõ ngày/phạm vi. Không viết lại metric lịch sử thành
metric mới; không coi tài liệu kế hoạch là bằng chứng đã triển khai.

Không sửa code sản phẩm, .env, checkpoint/corpus thật, scheduler hoặc model artifact.
Các thay đổi code có sẵn trong git status không thuộc lượt cập nhật tài liệu này.
Không kiểm lại live endpoint, crawling nguồn thật, model GPU/hash archive lớn hoặc
hiệu lực quy định giao dịch. Những mục này cần đánh giá riêng khi có yêu cầu.

## Các sai lệch đã xử lý

| Nội dung cũ | Đối chiếu và cập nhật | Nguồn code chính |
|---|---|---|
| Năm agent / sơ đồ thiếu Research, Forecast | Sáu role; forecast là service trước Research | agents/, graph/stock_agent_graph.py |
| Chỉ tái phân tích Technical | Ba action typed và optional approved evidence follow-up | Manager.review, evidence_follow_up.py |
| Sơ đồ veto rồi WATCH/HOLD | Veto chặn BUY và hiện dẫn AVOID; dừng retry nếu không có recoverable data issue | Manager._select_reanalysis_action/finalize |
| Tool allowlist như sandbox | AgentSpec là contract; LLM selection và state-update checks có phạm vi riêng | agent_contracts.py, agent_execution.py |
| Task plan chỉ là declaration | PlanExecutor/ChatPlanExecution kiểm actual artifacts/status/dependency | plan_executor.py, chat_plan_execution.py |
| NLU 120/122 câu, Qwen2.5 là default mới | 126 câu; Qwen3 client default, legacy endpoint vẫn tùy config | semantic_queries.json, llm_provider.py |
| RSI chart tách implementation | Chung TechnicalIndicatorToolkit | chart_service.py |
| Stale financial luôn UNKNOWN | Compatible current statements ưu tiên; stale ratio còn dùng được có quality weighting | FundamentalAgent.run, financial_facts.py |
| Provider default ba retry/0.25s | Provider một attempt, pacing 3.2s/1.05s; generic policy khác | market_provider.py |
| Timeout là hard limit | Cooperative deadline, clamp khi transport hỗ trợ | resilience.py, request_telemetry.py |
| Chưa PDF/version/source-policy | Text-layer PDF, supersedes, crawler source/path/policy đã có; chưa OCR | content_extractor.py, research_store.py, research_sync.py |
| Web chỉ discovery / discovery không bao giờ lưu | Fetch/extract external evidence; chat enqueue metadata có giới hạn, không tự ingest full-text | research_orchestrator.py, ResearchSyncService.enqueue_chat |
| Screening đòi ngày giống tuyệt đối | Tolerance mặc định ba ngày; ComparisonSpec có guard riêng | screening.py, screening_service.py |
| Deep default max(10, 2K) | Mặc định K, explicit caller override có giới hạn | RecommendationService.recommend |
| Chưa model trained / model nào cũng 20% | V2/V3/V4 adapters, weight/eligibility guards, V4 experimental opt-in | price_forecast.py, TechnicalAgent._apply_lstm_blend |
| CLI tự dùng model theo env | Bare AnalysisService trong CLI dùng baseline; UI có environment-aware service riêng | cli.py, streamlit_app.py |
| Mọi specialist review đúng một lần | Final-stage review một lần; approved follow-up có thể thêm review | stock_agent_graph.py, evidence_follow_up.py |
| Scheduler/.env/corpus được khẳng định là hiện trạng | Chuyển thành hướng dẫn hoặc bằng chứng lịch sử; không xác nhận live từ code | crawler guide, demo specialist guide |
| Tổng test từng phase như số hiện tại | Giữ số gốc và gắn phạm vi lịch sử; số audit mới ghi riêng | pytest log, docs/README.md |
| Nhiều agent được chứng minh tốt hơn một agent | Benchmark single_pass vẫn sáu role; chưa có baseline một LLM agent | coordination_benchmark.py |

## Kiểm chứng mới

| Kiểm tra | Kết quả | Phạm vi |
|---|---|---|
| Full pytest | **1023 passed, 89.74s** | Mock default, Neo4j off, local_hash, D-drive temp; không kiểm uptime live |
| Semantic evaluator | Intent/ticker **126/126**; period 6/6; follow-up 24/24; clarification 6/6; multi-intent 5/5; ECE **0.0507** | Author-created regression; không independent holdout |
| Frozen coordination | Cả hai graph 0 constraint violation; no-risk 2; bounded dùng 23 tool vs 12 single-pass, cùng 4 unresolved conflicts | Bốn synthetic snapshots; không đo lợi nhuận/accuracy đầu tư |
| Markdown local links | Không có đích bị thiếu trong README và docs | Chỉ kiểm file/đường dẫn local; không gọi URL ngoài hoặc kiểm nội dung trang live |

Log full suite: `.tmp/docs-audit-verified-20261023.log`; benchmark JSON:
`.tmp/docs-audit-coordination-20261023.json`. Đây là artifact local trong workspace,
không phải corpus người dùng hoặc report thị trường.

Lượt pytest đầu thiếu --basetemp bị dừng sau lỗi temp; lượt sandbox có --basetemp
trên D cũng gặp WinError 5 ở tmpdir và cleanup nên không có tổng hợp pass đáng tin.
Chạy lại cùng bộ test với quyền thực thi phù hợp hoàn tất 1023/1023. Không đổi
test hoặc logic để làm suite xanh. Python được gọi qua `.venv/Scripts/python.exe`;
không cài dependency vào system Python hoặc tải training package.

## Tài liệu nên dùng sau audit

1. [Báo cáo dự án hiện tại](project-report.md): phạm vi, kiến trúc, agent/tool/goal, dữ liệu/model/runtime và số đo mới.
2. [Bài nói specialization](agent-specialization.md): bản học thuyết trình và thứ tự mở code.
3. [Kiến trúc chi tiết](current-architecture.md), [kiến trúc luồng](architecture.md), [prompt contracts](agent-prompt-contracts.md).
4. [Danh mục báo cáo](README.md): phân biệt hiện hành, lịch sử, kế hoạch và model review.

Các review GPU V3/V4 và model card Qwen giữ nguyên kết luận/số đo đã ghi; audit
này cập nhật liên kết/trạng thái giữa báo cáo, không xác nhận lại nội dung artifact.
Các gate còn thiếu: real-data independent labels, forward validation, live latency,
OCR, point-in-time completeness và vận hành nhiều người dùng.
