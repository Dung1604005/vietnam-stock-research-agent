# Báo cáo hiện trạng dự án Stock Multi-Agent

Ngày đối chiếu: **23/10/2026**, theo working tree tại `D:\Download\thayque\Agent`.
Repository có sẵn nhiều thay đổi chưa commit; báo cáo phản ánh code trên đĩa,
không khẳng định đó là trạng thái HEAD hoặc bản đã triển khai. Lượt này cập nhật
tài liệu, không sửa logic sản phẩm, cấu hình riêng tư hoặc model artifact.

## 1. Mục đích và kết quả thực tế

Dự án là ứng dụng nghiên cứu cổ phiếu Việt Nam phục vụ học tập: chat Streamlit,
CLI, dữ liệu vnstock, chỉ báo/tài chính/rủi ro bằng Python, bằng chứng tài liệu và
LLM có giới hạn. Hệ thống không tích hợp broker/đặt lệnh. BUY/HOLD/WATCH/AVOID là
nhãn policy heuristic, confidence chưa calibrated và không phải xác suất sinh lời.

Có **sáu vai trò agent**, không phải năm: Data, Research, Technical, Fundamental,
Risk và Manager. Forecast, KnowledgeService, ChartService, RAG/store và router là
service/tool; không cộng chúng thành agent mới. Vai trò khác nhau có thể dùng chung
một provider/model LLM với schema/prompt riêng; không cần sáu model hoặc sáu call.

## 2. Phân vai, goal, tool và quyền quyết định

Mọi đường dẫn code trong bảng tính từ `src/stock_multi_agent/`.

| Agent | Goal thực tế | Công cụ/capability | Đầu ra và giới hạn | Code |
|---|---|---|---|---|
| Data | Lấy, kiểm tra, chuẩn hóa dữ liệu thật | MarketDataTool.get_stock_data, benchmark VNINDEX | MarketData/benchmark; giữ nguồn/ngày; không khuyến nghị, can_use_llm=False | [data_agent.py](../src/stock_multi_agent/agents/data_agent.py) |
| Research | Lấy evidence doanh nghiệp có citation | ResearchStore.search hybrid vector/FTS; bounded external retrieval; optional expanded review | research_evidence/freshness; không tự phát tín hiệu giao dịch | [research_agent.py](../src/stock_multi_agent/agents/research_agent.py) |
| Technical | Đánh giá xu hướng theo horizon | TechnicalIndicatorToolkit; tối đa hai diagnostic do LLM chọn; forecast blend đủ điều kiện | TechnicalAnalysis; Python sở hữu số, score và signal | [technical_agent.py](../src/stock_multi_agent/agents/technical_agent.py) |
| Fundamental | Đánh giá doanh nghiệp/định giá độc lập với giá | Metric/fact normalization, sector thresholds, quality weighting, FinancialToolkit, optional review/fact proposals | FundamentalAnalysis; không tự điền số thiếu hoặc để prose sửa score | [fundamental_agent.py](../src/stock_multi_agent/agents/fundamental_agent.py) |
| Risk | Phản biện, ngăn tự tin thiếu căn cứ | Volatility, drawdown, traded value, issuer/portfolio/event checks; optional evidence review | RiskAnalysis; tách rủi ro đo được/độ bất định; veto bằng rule | [risk_agent.py](../src/stock_multi_agent/agents/risk_agent.py) |
| Manager | Lập kế hoạch, phân công, xử lý conflict, kết luận | Typed semantic routing, dispatch/review, follow-up approval, grounded synthesis | FinalDecision bằng Python policy; không tính lại chỉ số hay bỏ veto | [manager_agent.py](../src/stock_multi_agent/agents/manager_agent.py) |

Các tên như `sector_template`, `quality_weighting` trong AgentSpec là capability
khai báo, không nhất thiết mỗi tên là một tool endpoint độc lập. FinancialToolkit
hiện có completeness/format helpers; scoring nằm ở FundamentalAgent và các service
facts. AgentSpec.allowed_tools không phải sandbox chặn mọi lời gọi Python.

[AgentSpec và context](../src/stock_multi_agent/models/agent_contracts.py),
[runtime authority](../src/stock_multi_agent/services/agent_execution.py) quy định
và kiểm tra output keys, sender, audit envelope. Chỉ Manager có quyền phát quyết
định cuối. Context chọn input theo vai trò nhưng không cô lập process; các agent
vẫn là code Python trong cùng ứng dụng, dùng shared AgentState.

## 3. Luồng thực thi và phối hợp

Full analysis: `Manager.plan → Data → Forecast → Research → dispatch →
Technical || Fundamental → join → Risk → Manager.review`.

Manager có ba hướng phân tích lại: extended Technical rồi Risk; Research rồi
Fundamental/Risk; hoặc refresh Data rồi forecast/research/các specialist/Risk.
Giới hạn mặc định hai vòng, còn phụ thuộc action history, bằng chứng có đổi hay
không và ngân sách. Veto không có vấn đề dữ liệu phục hồi được không được dùng
làm lý do lặp đến khi có BUY. Sau vòng conflict có optional Fundamental/Risk
review, follow-up do Manager duyệt và finalize. Follow-up có thể tạo thêm review
trên evidence mới, vì vậy không cam kết mỗi role chỉ gọi đúng một lần toàn request.

Code: [StockAgentGraph._build](../src/stock_multi_agent/graph/stock_agent_graph.py),
[Manager.review/finalize](../src/stock_multi_agent/agents/manager_agent.py),
[evidence_follow_up](../src/stock_multi_agent/services/evidence_follow_up.py).

Chat dùng HybridSemanticParser → IntentRouter/RoutingPlanValidator →
ChatPlanExecution/PlanExecutor. Executor gắn task được duyệt với output thực tế,
kiểm schema và trạng thái; full graph/batch vẫn là composite do service quản lý.
Đây không phải planner tự sinh code/công cụ tùy ý. Technical/chart/theory route
không bắt buộc chạy cả sáu vai trò; selective API có thể thực hiện stage tuần tự.

Code: [router](../src/stock_multi_agent/chat/router.py),
[chat execution](../src/stock_multi_agent/services/chat_plan_execution.py),
[executor](../src/stock_multi_agent/services/plan_executor.py),
[AnalysisService](../src/stock_multi_agent/services/analysis_service.py).

BUY yêu cầu Technical và Fundamental cùng BUY, Risk không HIGH, data uncertainty
không HIGH, không veto, evidence completeness từ 0.75 và không blocking conflict.
Veto dẫn đến AVOID trong policy hiện hành; không nên minh họa veto rồi kết luận
WATCH/HOLD như sơ đồ cũ. Confidence tổng hợp dùng cả ba specialist rồi giảm theo
conflict/coverage, không suy từ risk thấp thành confidence cao.

## 4. Dữ liệu, RAG và tri thức

| Thành phần | Đã có trong code | Giới hạn cần giữ rõ |
|---|---|---|
| Market | vnstock VCI mặc định, OHLCV/ratio/profile/statement/event tùy cấu hình, provenance, cache SQLite, raw gzip/hash, quality gates | KBS fallback vẫn cùng thư viện; không phải đối soát nguồn độc lập. Publication time/adjustment/corporate actions chưa đủ cho mọi trường |
| Financial facts | Period/unit/scale/currency/basis/consolidation checks; current compatible statements ưu tiên hơn stale ratio growth | Ratio cũ khác có thể giảm trọng số; không đúng khi nói mọi stale financial flag luôn làm UNKNOWN |
| Indicators/charts | Chung TechnicalIndicatorToolkit; horizon profile, chart warm-up, diagnostics | Score/threshold chưa được chứng minh bằng outcome đầu tư độc lập |
| Local research | SQLite vector float32 + FTS; period/freshness gate; citation-constrained synthesis; document version/supersedes | local_hash là baseline n-gram, không phải neural embedding; citation hợp lệ không tự chứng minh claim đúng |
| External evidence | Bounded plan/discovery/fetch/extract, HTML và PDF có text layer qua optional pypdf | Snippet/link chưa fetch không phải evidence; chưa OCR/table extraction toàn diện |
| Crawler | Queue, source policy/domain/path, robots, permission review, incremental ingest, version/retry/report, script Scheduler | Có script không chứng minh Scheduler đang chạy. Chat discovery chỉ enqueue metadata, không tự cấp phép full-text |
| Theory KB | KnowledgeService deterministic, SQLite FTS, provenance, freshness warning; MIXED ghép metric thật | Không phải ResearchAgent hoặc nguồn quyết định BUY/SELL; freshness warning không tự xác nhận hiệu lực quy định |
| Graph | KB Neo4j projection; Investment Graph V2 có ledger/provenance, SQLite traversal và Graph-RAG opt-in | Projection không tạo fact độc lập; chưa có relational holdout độc lập chứng minh cải thiện |

Code chính: [market_provider](../src/stock_multi_agent/providers/market_provider.py),
[financial_facts](../src/stock_multi_agent/services/financial_facts.py),
[chart_service](../src/stock_multi_agent/services/chart_service.py),
[research_store](../src/stock_multi_agent/services/research_store.py),
[content_extractor](../src/stock_multi_agent/services/content_extractor.py),
[research_sync](../src/stock_multi_agent/services/research_sync.py).

Provider thực tế mặc định một attempt; pacing 3.2 giây khi chưa có key, 1.05 giây
khi có key, có thể cấu hình. Không lấy generic ResiliencePolicy mặc định ba
attempt/0.25 giây để báo cáo default thực tế của VnstockMarketDataProvider.

## 5. Forecast và LLM

V2/V3/V4 adapters đã có. UI chọn V2/V4 và dùng deferred environment-aware forecast;
bare AnalysisService(), bao gồm CLI hiện tại, mặc định inject PriceForecastService()
baseline. Chỉ có biến môi trường LSTM không tự làm mọi construction path load model.

Forecast hợp lệ/đủ policy có thể đóng góp tối đa 20% Technical; baseline/shadow
không có trọng số. V4 experimental opt-in là ngoại lệ có kiểm tra file/metadata,
không biến `policy_eligibility=false` thành model đã promote. Các báo cáo GPU
V3/V4 trước đây là bằng chứng development lịch sử; lượt audit này không retrain,
rehash toàn bộ artifact hoặc xác nhận forward-market performance.

LLM dùng chung adapter Colab Qwen/OpenAI/Ollama tùy config. Colab client default
là Qwen3; endpoint Qwen2.5 cũ không được tự đổi model. Data không gọi LLM;
Technical chỉ chọn tool allowlist; Fundamental/Risk review evidence; Manager chọn
typed intent/tổng hợp khi bật. Python vẫn giữ số, score, veto và final policy.
Structured schema, numeric/source checks và claim anchors có phạm vi, không bảo
đảm semantic truth. Qwen LoRA notebook/registry không đồng nghĩa adapter đã train
hoặc được duyệt phục vụ app.

Code: [forecast](../src/stock_multi_agent/services/price_forecast.py),
[CLI](../src/stock_multi_agent/cli.py),
[UI](../app/streamlit_app.py), [advisor](../src/stock_multi_agent/services/agent_llm.py),
[gateway](../src/stock_multi_agent/services/llm_gateway.py).

## 6. Screening, recommendation và vận hành

Default watchlist 100 mã; explicit current exchange universe vẫn bị chặn số mã
đánh giá, không phải universe lịch sử. Screening cho phép price-date lag mặc định
ba ngày, phân biệt exact alignment với comparability; ComparisonSpec có guard
riêng. Recommendation quick screen không LLM/LSTM từng mã, không refresh hàng
loạt, deep limit mặc định bằng K, không còn mặc định max(10, 2K). Có qualified
buckets và REFERENCE_ONLY công khai policy gaps; không bảo đảm luôn có K mã đạt chuẩn.

STANDARD giữ budget theo intent; QUICK 0 LLM/80 tool/30s, DEEP tối đa
4 LLM/250 tool/180s, OFFLINE 0 LLM/250 tool/120s và chặn outbound. Profile siết
budget intent, không tự nâng budget. Deadline cooperative; native/synchronous I/O
không hỗ trợ cancel có thể trả về muộn, không có hard wall-clock guarantee.

Graph checkpoint/pending writes và UI conversation được lưu SQLite trong
.runtime/runs. Resume tiếp tục node đang chờ, giữ profile và thu hẹp theo caller;
run đã xong không tự retry UNAVAILABLE. Không bảo đảm exactly-once external I/O,
khôi phục mọi UI state, auth/user isolation/retention/migration/backup production.

[Screening](../src/stock_multi_agent/services/screening_service.py),
[recommendation](../src/stock_multi_agent/services/recommendation_service.py),
[profiles](../src/stock_multi_agent/services/execution_profiles.py),
[checkpoint](../src/stock_multi_agent/services/run_checkpoints.py),
[conversation](../src/stock_multi_agent/chat/conversation.py).

## 7. Kiểm chứng ngày 23/10/2026

NLU evaluator chạy lại: intent **126/126**, ticker **126/126**, period **6/6**,
follow-up **24/24**, clarification **6/6**, multi-intent **5/5**; ECE **0.0507**.
Đây là regression do dự án tạo, không phải độc lập/traffic thật hoặc xác nhận
accuracy production. Mốc 120/122 trong tài liệu cũ là dataset lịch sử khác mẫu.

Benchmark `synthetic-frozen-coordination-v1`, bốn snapshot tổng hợp, cùng input/budget:

| Biến thể | Constraint violations | UNKNOWN specialist signals | Conflict chưa giải quyết | Tool calls | LLM calls |
|---|---:|---:|---:|---:|---:|
| single_pass | 0 | 1 | 4 | 12 | 0 |
| bounded_coordination | 0 | 1 | 4 | 23 | 0 |
| no_risk_baseline | 2 | 1 | Không đo bởi critic | 12 | 0 |

Single-pass vẫn là graph sáu vai trò, chỉ đặt max_iterations=0; **không phải một
LLM agent duy nhất**. No-risk là ablation chỉ trong evaluator. Thêm vòng không
giảm conflict trên snapshot đóng băng và tăng tool calls. Kết quả không chứng minh
multi-agent chính xác/nhanh/sinh lời hơn single-agent trên thực tế.

Full pytest chạy lại: **1.023 passed trong 89,74 giây**; chi tiết môi trường và
lỗi quyền temp của lượt đầu được ghi ở [bản ghi audit](documentation-audit-2026-10-23.md).
Mốc 1.023 test ngày 22/10 và các mốc phase nhỏ hơn được giữ là kết quả lịch sử,
không tự nhận thành kết quả lần chạy mới.

Lệnh tái lập (PowerShell, tất cả temp trong D workspace):

```powershell
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:STOCK_AGENT_LLM_PROVIDER = 'mock'
$env:STOCK_AGENT_KNOWLEDGE_GRAPH_ENABLED = 'false'
$env:STOCK_AGENT_RAG_EMBEDDING_PROVIDER = 'local_hash'
.\.venv\Scripts\python.exe -m pytest --basetemp .tmp/docs-audit-recheck --tb=short
.\.venv\Scripts\python.exe scripts\evaluate_semantics.py --show-errors
.\.venv\Scripts\python.exe scripts\evaluate_coordination.py --output .tmp/coordination-recheck.json
```

Không gọi lại live vnstock/Colab/web, không kiểm Scheduler, không thay corpus,
không train GPU, không dùng software pass để khẳng định model gate hoặc production readiness.

## 8. Cách dùng báo cáo để thuyết trình

Đọc [bài nói về specialization](agent-specialization.md), mở AgentSpec rồi graph,
tiếp theo runtime OUTPUTS/validate_update và Manager.finalize để chứng minh ranh
giới trách nhiệm thực sự. Nhấn mạnh lợi ích là kiểm soát quyền, kiểm chứng từng
bước, thay module và phối hợp song song có điều kiện; đổi lại thêm orchestration,
state, retry/latency và call budget. Các vai trò dùng chung dữ liệu/model vẫn có
thể mắc lỗi tương quan; sự hiện diện của Risk không thay thế dữ liệu độc lập.
