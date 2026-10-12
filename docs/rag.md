# RAG với vector similarity

Luồng research cục bộ hiện chạy end-to-end:

```text
Câu hỏi
  -> EmbeddingProvider
  -> cosine similarity trên research_chunk_embeddings
  -> hybrid merge với SQLite FTS5
  -> top-k chunk có provenance
  -> Ollama/OpenAI structured synthesis
  -> kiểm tra source_id
  -> câu trả lời kèm citation
```

`ResearchStore` lưu vector dạng normalized `float32` trong SQLite, gắn với `chunk_id` và `model_id`. Corpus cũ được backfill tự động khi tìm kiếm lần đầu; có thể chủ động chạy:

```powershell
.\.venv\Scripts\python.exe scripts\reindex_research_vectors.py --force
```

Mặc định `local_hash` tạo embedding offline 768 chiều bằng word/character n-gram và một số alias tài chính Việt–Anh. Đây là baseline miễn phí, không phải neural semantic model. Nếu đã cài model embedding trong Ollama, cấu hình:

```text
STOCK_AGENT_RAG_EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

LLM chỉ nhận tối đa `STOCK_AGENT_RAG_TOP_K` chunk đã truy xuất. Mỗi claim phải dùng `S1..Sk`; source ID không tồn tại bị loại và toàn bộ synthesis bị từ chối nếu không còn claim hợp lệ. Nội dung tài liệu được coi là untrusted data để giảm prompt injection. Khi embedding hoặc LLM lỗi, hệ thống giữ FTS/citation response thay vì bịa câu trả lời.

Giới hạn hiện tại: corpus research còn nhỏ, chưa có benchmark Recall@k/MRR/faithfulness đủ lớn, chưa OCR PDF scan và `local_hash` không hiểu ngữ nghĩa sâu như model embedding đã huấn luyện.
