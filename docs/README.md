# Danh mục tài liệu và báo cáo dự án

Kiểm kê ngày **23/10/2026**. Nguồn đọc chính là [báo cáo hiện trạng](project-report.md),
[kiến trúc chi tiết](current-architecture.md) và [bài thuyết trình phân vai](agent-specialization.md).
Chi tiết sai lệch đã sửa và kiểm chứng mới: [audit tài liệu](documentation-audit-2026-10-23.md).

## Quy tắc đọc báo cáo

- Working tree/code và kết quả chạy mới xác định hiện trạng. Một tài liệu có ngày mới không tự xác nhận live endpoint, artifact, corpus hoặc Scheduler.
- Báo cáo theo phase giữ số test, smoke test, corpus và metric của lượt triển khai gốc; không cộng chúng thành tổng test hiện tại.
- Roadmap/kế hoạch có cả phần khảo sát trước triển khai và tiêu chí mục tiêu. Các gate còn mở không được đánh dấu hoàn tất chỉ vì code đã có.
- Review model phân biệt train hoàn tất, development gate, promotion và forward validation; V4 experimental UI không phải promotion.
- Khi trích dẫn code, dùng file + tên class/hàm; line number có thể đổi sau cập nhật repository.

## Kết quả chạy lại

Full suite **1023 passed / 89.74s**; NLU intent/ticker **126/126**, ECE **0.0507**.
Benchmark bốn snapshot tổng hợp: 0 violation cho single-pass và bounded graph,
2 violation cho evaluator-only no-risk. Single-pass vẫn sáu role; chưa có baseline
một LLM agent. Mọi kết quả này là kiểm chứng engineering, không accuracy đầu tư.

## Danh mục toàn bộ Markdown trong docs

| Tài liệu | Loại | Cách sử dụng / giới hạn |
|---|---|---|
| [Prompt contract agent-evidence-v3-claim-anchors](agent-prompt-contracts.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Bài thuyết trình: chuyên môn hóa vai trò agent và code tương ứng](agent-specialization.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Architecture](architecture.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Crawler và auto-ingestion — triển khai và sử dụng](automatic-research-crawler-implementation.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Kế hoạch crawler và auto-ingestion cho Research RAG](automatic-research-crawler-plan.md) | Kế hoạch / mục tiêu | Checkbox và mô tả đích không tự chứng minh đã triển khai |
| [Kế hoạch tăng tính chủ động cho hệ agent chứng khoán](autonomous-agents-plan.md) | Kế hoạch / mục tiêu | Checkbox và mô tả đích không tự chứng minh đã triển khai |
| [Conversational assistant architecture](chat-architecture.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Qwen2.5-3B-Instruct trên Google Colab](colab-qwen-setup.md) | Hướng dẫn legacy | Qwen2.5 cũ; Qwen3 xem qwen3-agent-setup |
| [Kế hoạch nâng cấp toàn diện hệ thống Multi-Agent (ưu tiên công cụ miễn phí)](comprehensive-free-upgrade-plan.md) | Kế hoạch / mục tiêu | Checkbox và mô tả đích không tự chứng minh đã triển khai |
| [Conflict resolution between agents](conflict-resolution.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Kiến trúc hiện tại](current-architecture.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Demo: biểu đồ, giải thích và LLM Fundamental/Risk](demo-analysis-charts-specialists.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Demo Phase 8: chạy thật, replay rõ nguồn, không gọi dịch vụ ngoài khi offline](demo-runbook.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Audit và cập nhật báo cáo — 23/10/2026](documentation-audit-2026-10-23.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Kiến trúc Dual Recommendation](dual_recommendation_proposal.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [GitHub publishing checklist](github-publish.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Implementation baseline and remaining work](implementation-plan.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Investment Graph V2 — ontology và provenance](investment-graph-v2.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Chạy LSTM V3 trên Kaggle sau khi bổ sung `.zip.bin`](kaggle-v3-ready-notebook.md) | Reproduction / experiment | Pipeline/gate riêng; không mặc nhiên model đã promote |
| [Knowledge Base chứng khoán](knowledge-base.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Knowledge Graph với Neo4j](knowledge-graph.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Qwen3-4B LoRA V1 — model card trước training](llm-adapter-model-card.md) | Model card / review artifact | Kết luận theo artifact/lượt review; không retrain/rehash trong audit |
| [Fine-tuning Qwen3 trên Google Colab — Phase 7](llm-finetuning-data-policy.md) | Reproduction / experiment | Pipeline/gate riêng; không mặc nhiên model đã promote |
| [Tự mở website và tương tác trên localhost](local-interactions.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Workspace tương tác trên localhost](local-workspace.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [LSTM training on Kaggle](lstm-kaggle-training.md) | Reproduction / experiment | Pipeline/gate riêng; không mặc nhiên model đã promote |
| [LSTM V4.1 development experiment](lstm-kaggle-v4-1.md) | Reproduction / experiment | Pipeline/gate riêng; không mặc nhiên model đã promote |
| [LSTM V4: leakage-aware Kaggle experiment](lstm-kaggle-v4.md) | Reproduction / experiment | Pipeline/gate riêng; không mặc nhiên model đã promote |
| [Chọn LSTM V2 / V4 trên UI](lstm-ui-model-selection.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [LSTM V3: engineering pipeline and honest model status](lstm-v3-model-card.md) | Model card / review artifact | Kết luận theo artifact/lượt review; không retrain/rehash trong audit |
| [Sửa gate số học và đánh giá lại kết quả LSTM V3](lstm-v3-numerical-gate-review.md) | Model card / review artifact | Kết luận theo artifact/lượt review; không retrain/rehash trong audit |
| [LSTM V4 — điều tra V3 và ưu tiên dự báo phiên kế tiếp](lstm-v4-research-and-training.md) | Model card / review artifact | Kết luận theo artifact/lượt review; không retrain/rehash trong audit |
| [Multi-agent reliability và giới hạn vận hành](multi-agent-reliability.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Lọc cổ phiếu bằng điều kiện số](numeric-screening.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Nâng cấp Phase 0, 1, 2 — triển khai và kiểm chứng](phase-0-2-upgrade-implementation.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Kế hoạch khắc phục các hạn chế Phase 3–5](phase-3-5-reliability-remediation.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Triển khai Phase 3–5](phase-3-5-upgrade-implementation.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Kế hoạch triển khai Phase 6–8 — LSTM, fine-tuning và demo](phase-6-8-execution-plan.md) | Kế hoạch / mục tiêu | Checkbox và mô tả đích không tự chứng minh đã triển khai |
| [Triển khai Phase 6–8 và cách chạy](phase-6-8-implementation.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Phase A — Reliable Data Layer](phase-a-data-layer.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Phase B — Semantic Understanding & Dynamic Routing](phase-b-semantic-routing.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Phase C — Internet Research Audit](phase-c-internet-research-audit.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Phase C — Research, Retrieval & Evidence](phase-c-research-evidence.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Phase D — Analysis Engine Baseline](phase-d-analysis-engine.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Phase E — Screening and Comparison](phase-e-screening.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Phase F — Evaluation and Point-in-Time Backtesting](phase-f-evaluation-backtesting.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Price forecasting: implemented models and policy](price-forecasting.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Product roadmap — AI Investment Research Assistant](product-roadmap.md) | Kế hoạch / mục tiêu | Checkbox và mô tả đích không tự chứng minh đã triển khai |
| [Báo cáo hiện trạng dự án Stock Multi-Agent](project-report.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Kết nối các agent với Qwen3](qwen3-agent-setup.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [RAG với vector similarity](rag.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Kỳ báo cáo và kiểm tra độ mới của Research RAG](research-data-freshness.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Semantic NLU baseline](semantic-baseline.md) | Báo cáo lịch sử | Giữ số liệu/mốc gốc; hiện trạng xem project-report |
| [Hybrid semantic NLU](semantic-nlu.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |
| [Vnstock Community API key](vnstock-community-api.md) | Hiện trạng / hướng dẫn | Code có thể cấu hình; không đồng nghĩa dịch vụ/corpus live đã kiểm lại |

README gốc là cổng chạy/setup và liên kết tổng quan. Hướng dẫn thao tác cục bộ,
API key, publish hoặc dịch vụ bên ngoài không xác nhận trạng thái đang chạy trên
máy chỉ từ sự tồn tại của script. Các tài liệu không cần sửa nội dung trong audit
vẫn được phân loại ở bảng để tránh nhầm phạm vi.
