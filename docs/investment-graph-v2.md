# Investment Graph V2 — ontology và provenance

Version: `investment-graph-v2`. Ba lớp không bị trộn:

- `theory`: nguồn Theory KB và topics; không dùng làm tin tức hiện tại.
- `research`: document/chunk/ticker/kỳ báo cáo; không lấy ngày đăng làm kỳ dữ liệu.
- `structured`: profile, quan sát tài chính, issuer events và model output có
  source reference. Forecast vẫn là đầu ra mô hình, không phải giá thực tế.

Schema node: Company, Ticker, Exchange, Sector, Person, Document, Event, Metric,
Period, Topic, RiskFactor. Schema edge: LISTED_AS, IN_SECTOR, PUBLISHED,
MENTIONS, REPORTS_METRIC, HAS_EVENT, EXPOSED_TO, RELATED_TO, SUPERSEDES.

## Mapping thực hiện

| Input | Projection |
|---|---|
| ResearchDocument + chunks | Document MENTIONS Ticker/Period; SUPERSEDES bản cũ |
| KnowledgeDocument + topics | Document MENTIONS Topic |
| Profile có source URL | Company LISTED_AS Ticker, IN_SECTOR Sector |
| Financial observation có kỳ và source URL | Document REPORTS_METRIC Metric, MENTIONS Ticker/Period |
| Issuer event có source URL | Ticker MENTIONS Event |
| Forecast + source OHLCV | Ticker RELATED_TO Metric, evidence ghi model/version/estimate |

Không tự suy ra `PUBLISHED`, sở hữu, lãnh đạo, tác động sự kiện hay rủi ro chỉ từ
ticker tag hoặc đồng xuất hiện. Các quan hệ đó được schema hỗ trợ nhưng cần
triple có nguồn và review. Không dùng LLM để tạo số liệu thiếu.

`ig_evidence` giữ URL/quote/document/layer/published_at/retrieved_at. `ig_edges`
giữ evidence_id/valid_from/valid_to/confidence/payload. `ig_node_evidence` giữ
lineage theo thời gian cho node dùng chung; không ghi đè mọi nguồn thành một
"sự thật hiện tại". Neo4j lưu RELATES_V2 riêng với legacy KB; edge giữ đầy đủ
provenance, node properties chỉ là projection gần nhất, ledger SQLite mới là
nguồn để truy toàn bộ provenance.

Ngày công bố không biết được lưu null. Khi đó thời điểm biết claim là lúc tải,
không tự backdate. Alias có thể trả nhiều entity ID; tên trùng không tự merge.
Đổi ticker có thể biểu diễn bằng triple reviewed với validity interval và alias
có nguồn, không có registry đổi tên giả tạo sẵn.

## Đồng bộ và truy vấn

```powershell
.\.venv\Scripts\python.exe scripts\sync_investment_graph.py
.\.venv\Scripts\python.exe scripts\sync_investment_graph.py --neo4j
.\.venv\Scripts\python.exe scripts\sync_investment_graph.py --query "Nguồn nào nhắc tới FPT?"
```

Không có Neo4j vẫn đọc/traverse 1–2 hop bằng SQLite. MERGE theo ID quan hệ có
provenance nên chạy lại không nhân bản. Không có thao tác xóa/rebuild graph
hiện hữu; namespace mới không đụng legacy graph. Fingerprint bỏ qua document
không đổi. Tài liệu mới ingest được projection tự động, lỗi projection không
làm mất document đã commit.

Graph-RAG mới chỉ bật khi `STOCK_AGENT_GRAPH_RAG_V2_ENABLED=true`. Candidate từ
graph vẫn qua ticker/date/period/freshness gate và citation audit trước synthesis.
20 test relational synthetic không thay thế 20 câu hỏi/corpus độc lập. Chưa có
phép đo Recall@5 độc lập tăng ≥5 điểm phần trăm, vì thế default vẫn là hybrid
vector/FTS, không tự bật graph để làm đẹp kiến trúc.
