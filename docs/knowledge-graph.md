# Knowledge Graph với Neo4j

## Vai trò

Knowledge Graph là lớp quan hệ bổ sung cho Knowledge Base và RAG, không thay thế SQLite/FTS5 hay vector similarity.

```text
Câu hỏi
  -> Exact/BM25/vector retrieval
  -> nhận dạng entity trong Neo4j
  -> traversal tối đa 2 hop
  -> lấy document_id liên quan
  -> hợp nhất candidate bằng RRF
  -> câu trả lời có chunk citation
```

KB SQLite vẫn là source of truth cho nội dung, URL, timestamp và chunk. Neo4j chỉ lưu graph projection; nếu Neo4j dừng, truy xuất KB hiện tại vẫn hoạt động.

## Schema graph

Mọi node có label `KnowledgeEntity`, thuộc namespace `stock-kb`, và có thuộc tính `type`:

- `DOCUMENT`: tài liệu trong KB;
- `TOPIC`: chủ đề từ manifest;
- `CONCEPT`: P/E, ROE, RSI, DCF, thanh khoản, drawdown…;
- `SECTION`: heading/chunk có provenance;
- `PUBLISHER`: đơn vị phát hành.

Quan hệ dùng type Neo4j `RELATES` và thuộc tính `kind`: `HAS_TOPIC`, `MENTIONS`, `HAS_SECTION`, `PUBLISHED_BY`, `ABOUT_TOPIC`. Quan hệ giữ `evidence_document_id` và `evidence_chunk_id` để traversal không làm mất provenance.

## Cài đặt cục bộ trên ổ D

Project dùng Neo4j Community 2026.09.0 và Python driver 6.x. Neo4j được cài và chạy trực tiếp trong `.runtime/neo4j` của workspace trên ổ D; không cần ánh xạ ổ đĩa phụ.

```powershell
.\scripts\setup_neo4j.ps1
.\scripts\start_neo4j.ps1
.\.venv\Scripts\python.exe scripts\sync_knowledge_graph.py --rebuild
```

Workspace nên dùng đường dẫn ASCII như `D:\Download\thayque\Agent` để Java/Neo4j xử lý classpath ổn định.

Neo4j Browser: `http://localhost:7474`; Bolt: `bolt://localhost:7687`.

Cấu hình `.env`:

```text
STOCK_AGENT_KNOWLEDGE_GRAPH_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<local password>
NEO4J_DATABASE=neo4j
```

Không commit mật khẩu thật. Khi deploy, dùng Streamlit Secrets hoặc biến môi trường và một Neo4j/AuraDB instance có thể truy cập từ cloud; `localhost` của máy cá nhân không truy cập được từ Streamlit Community Cloud.

## Đồng bộ và truy vấn

Ingest KB mới sẽ in trạng thái Neo4j sync. Có thể chủ động đồng bộ lại:

```powershell
.\.venv\Scripts\python.exe scripts\sync_knowledge_graph.py
.\.venv\Scripts\python.exe scripts\sync_knowledge_graph.py --rebuild
```

Ví dụ kiểm tra trong Neo4j Browser:

```cypher
MATCH p=(concept:KnowledgeEntity {id: 'concept:pe'})-[:RELATES*1..2]-(document:KnowledgeEntity {type: 'DOCUMENT'})
RETURN p
LIMIT 25;
```

## Giới hạn

- Entity extraction hiện deterministic theo taxonomy/alias tài chính, không dùng LLM để tự tạo fact.
- Traversal giới hạn hai hop nhằm tránh context lan quá rộng.
- Neo4j không được phép tạo BUY/SELL; nó chỉ mở rộng tài liệu ứng viên.
- Fact và citation cuối cùng vẫn phải quay về chunk trong KB SQLite.

