# Qwen3-4B LoRA V1 — model card trước training

Ngày18/10/2026. Trạng thái **UNTRAINED / GPU_SMOKE_PENDING / NOT_APPROVED**. Không có số đo chất lượng hoặc tốc độ GPU được tuyên bố.

| Thuộc tính | Giá trị |
| --- | --- |
| Base | `Qwen/Qwen3-4B-Instruct-2507` |
| Base/tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| License | Apache-2.0; phải review quyền dataset riêng |
| Notebook | `notebooks/finetune_qwen3_4b_lora_v1.ipynb` |
| Phương pháp | QLoRA NF4 4-bit, double quantization, FP16 compute, rank8/alpha16 |
| LoRA modules | q/k/v/o projections, dropout0.05 |
| Context/batch | 512token ban đầu, microbatch1, accumulation8 |
| Loss | Assistant-answer-only; không truncate nhãn để ép fit |
| Training | Pilot50steps validation-only; full2epochs, validation-loss checkpoint selection |
| Task khởi đầu | NLU JSON một task; schema pilot không phải schema NLU production |
| Dữ liệu mặc định | Chưa có human-reviewed dataset; authored synthetic demo phải chọn rõ ràng |
| Quyền tác động | Không thay score, forecast, validator, broker policy hoặc LSTM V2 |

## Trước khi công bố adapter

Điền từ artifact thật: dataset digest/status và nguồn/quyền; adapter weight/config hashes; prompt/schema/chat-template hashes; code/config revision; library/CUDA/GPU identity; dropped-length counts; throughput và peak memory; full raw A/B metrics/CI; lỗi theo nhóm; ví dụ abstention; reviewer/license approval; limitations và rollback.

Không lấy training loss làm bằng chứng model tốt hơn. Không dùng test đã xem nhiều lần làm independent holdout. Synthetic draft chỉ phục vụ pipeline/demo, không promote dù vượt baseline. Exact-match/anchor checks không chứng minh chân lý tài chính; synthesis cần blind human review.

## Triển khai

Gói là PEFT adapter, không phải full4B base model; inference vẫn phải tải đúng base revision. Model/card/code không tự upload weights lên Hugging Face/GitHub hoặc bật app endpoint. Registry kiểm hash/approval, client kiểm identity; serving sai task/prompt/schema bị từ chối. Giữ base+RAG làm đối chứng và rollback.

Hướng dẫn chạy và đánh giá: [llm-finetuning-data-policy.md](llm-finetuning-data-policy.md). Nguồn: [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
