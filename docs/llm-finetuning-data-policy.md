# Fine-tuning Qwen3 trên Google Colab — Phase 7

Ngày cập nhật: 18/10/2026. **Code và notebook đã chuẩn bị; chưa train GPU, chưa có adapter đạt gate.** Không cần API trả phí. Không cài Torch/PEFT vào `.venv` ứng dụng.

## 1. Mục tiêu và giới hạn

Model là `Qwen/Qwen3-4B-Instruct-2507`, revision cố định `cdbee75f17c01a7cc42f958dc650907174af0554`. Một base model dùng chung cho các vai trò; adapter ban đầu chỉ chuyên **một task**. Fine-tuning dạy format/đọc yêu cầu/trích xuất có cấu trúc, không dạy ghi nhớ giá mới nhất, không thay RAG, không trao phép tính deterministic hoặc quyền đặt lệnh cho LLM.

Model này là bản non-thinking, license Apache-2.0. Notebook dùng QLoRA NF4 4-bit, FP16, rank8, context512, microbatch1, accumulation8; đây là cấu hình ứng viên cho T4, **không phải kết quả đã đo trên GPU**. [Model card/license chính thức](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).

Colab free không bảo đảm GPU hoặc thời lượng phiên. Quy trình này dành cho training/batch evaluation/inference tương tác trong notebook; không cài tunnel, keepalive hay tự xuất bản endpoint. Xem [Colab FAQ](https://research.google.com/colaboratory/faq.html) trước khi triển khai dịch vụ.

## 2. Cách nhanh nhất để chạy thử trên Colab

1. Trên Colab chọn **File → Upload notebook** và upload `notebooks/finetune_qwen3_4b_lora_v1.ipynb`. Không copy JSON của `.ipynb` vào code cell: `null` trong JSON không phải Python.
2. Chọn **Runtime → Change runtime type → T4 GPU**. Chạy cell cài dependency; nếu đã import library trước đó, restart session rồi chạy lại từ cell định nghĩa, không cần cài lại mỗi lần.
3. Đọc license/quyền dữ liệu và chỉ đổi `UPLOAD_RIGHTS_CONFIRMED=True`, `LICENSE_REVIEWED=True` sau khi tự kiểm tra.
4. Nếu chưa có dữ liệu đã duyệt: chọn `ALLOW_SYNTHETIC_DEMO=True`. Notebook tự tạo 288 ví dụ **tự biên soạn, chưa duyệt**, chia theo template. Không đọc `.env`, chat history hay báo cáo riêng tư. Mode này chỉ kiểm tra pipeline; adapter luôn không đủ điều kiện promote.
5. Giữ `TRAIN_FULL=False`, `MAX_SEQUENCE_LENGTH=512`: pilot50steps, chỉ đo validation. Quan sát loss, thời gian và peak GPU. Không hứa train lâu hơn sẽ tốt hơn.
6. Chốt protocol rồi mở **output folder mới**, đặt `TRAIN_FULL=True` để chạy2epochs. Full run mới đo test; nếu tiếp tục tối ưu theo kết quả test đó thì nó đã thành development set, không còn holdout độc lập.
7. Cell cuối tải archive kết quả. Trên Windows lưu vào `D:\Download\thayque\Agent\.runtime\llm_training\results\`; không lưu vào C nếu không bắt buộc.
8. Cell `run_saved_inference(...)` cho thử prompt riêng sau khi reload trọng số đã lưu. Nó trả raw output với nhãn `EXPERIMENTAL_NOT_APPROVED`, không tự bật adapter trong ứng dụng.

Mỗi run có checkpoint25steps. Để chống mất phiên, có thể đặt `OUTPUT` trên Google Drive đã mount bằng thao tác rõ ràng của bạn; chỉ khi quyền upload cho phép. Tiếp tục run bị ngắt bằng `RESUME='auto'` hoặc checkpoint path và **giữ nguyên dataset/config/output folder**. Nếu đổi protocol, code từ chối resume. Run hoàn tất không được chạy lại vào cùng folder. Model download vẫn cần Internet và dung lượng trên filesystem Colab.

## 3. Tạo dataset trên D từ dữ liệu đã review

Pipeline chỉ nhận file JSONL được chỉ định, không quét hội thoại tự động. Mỗi dòng có `messages` kết thúc ở user; nhãn tách riêng `approved_output`; `output_schema`, task/prompt/schema version; template/document/conversation group; checksum tài liệu; rights/consent/cloud permission và human approval.

Các điều kiện bắt buộc cho dữ liệu thật:

- `usage_rights` là `AUTHORED`, `PUBLIC_DOMAIN`, hoặc `PERMISSION_GRANTED` kèm reference.
- `consent_to_training=true`; muốn tạo gói upload thì từng dòng phải có `cloud_upload_allowed=true`.
- `approval_status='APPROVED'`, `label_origin='human_reviewed'`, reviewer và reviewed_at có timezone. Đây là attestation do curator chịu trách nhiệm, không phải xác thực danh tính tự động.
- Không credentials, email/điện thoại/đường dẫn riêng hay portfolio cá nhân. Bộ lọc chỉ giảm rủi ro, không thay review. Redaction làm mất approval, phải review lại nhãn sau khi sửa.
- Giữ mọi paraphrase cùng template, hội thoại và phiên bản tài liệu trong một connected group. Không tự đổi group để ép chia được train/test. Issuer overlap được báo riêng; không tuyên bố đây là split đồng thời unseen issuer và unseen template.

Ví dụ tạo nháp local để review, chưa upload:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_llm_finetuning_dataset.py --generate-drafts 512 --output .runtime\llm_training\drafts_v1
```

Tạo gói từ JSONL đã duyệt (không tự upload):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_llm_finetuning_dataset.py --input .runtime\llm_training\reviewed_examples.jsonl --output .runtime\llm_training\reviewed_v1 --cloud-upload-opt-in
```

Chỉ để smoke test local nháp, có thể thêm `--allow-synthetic-demo` cùng `--generate-drafts` và `--cloud-upload-opt-in` sau khi đã preview. Gói `notebook-upload.zip` gồm đúng5file: manifest, schema, train/validation/test JSONL. Upload zip tại cell dữ liệu trên Colab với `ALLOW_SYNTHETIC_DEMO=False` cho dữ liệu đã duyệt, hoặc True nếu chính bạn chọn gói nháp. Hash lock không cho overwrite phiên bản dataset cũ.

**Schema pilot không phải schema app:** nháp có intent như `CHART`, `KNOWLEDGE`, trường `count/horizon/clarify`; app dùng `STOCK_CHART`, `THEORY`, `requested_result_count/...`. Không được dùng adapter pilot thay trực tiếp NLU app. Muốn triển khai, curation phải sử dụng chính system prompt, JSON schema Pydantic và task đang chạy; registry/gateway kiểm hash. Validator local hỗ trợ `$defs/$ref`, `anyOf/oneOf/allOf`, enum, nullable và date/time thường dùng; keyword không hỗ trợ bị từ chối, không silently bỏ qua.

## 4. A/B, gate và artifact

Full run tạo:

- `adapter/adapter_model.safetensors`, `adapter_config.json`, tokenizer files và `adapter-manifest.json` PENDING.
- `baseline-predictions.jsonl`, `candidate-predictions.jsonl`: raw generations, không deterministic fallback.
- `identities.json`: cùng base/tokenizer revision, quantization, decoding/prompt/schema/chat-template hashes; candidate thêm digest đúng trọng số/config.
- `experiment-lock.json`, `experiment-summary.json`, checkpoints và `reload-smoke.json`.
- `dataset/`: bản nguyên vẹn5file frozen để đánh giá/tái lập sau khi phiên Colab kết thúc. Archive kết quả là **private**, không tự public; kiểm tra quyền trước khi chia sẻ.

Pilot chỉ xuất `baseline-validation-predictions.jsonl` và `candidate-validation-predictions.jsonl`, **không dùng làm A/B test promotion**. Baseline validation gần trần trên dữ liệu thật mà thiếu error analysis sẽ dừng `NO_TRAINING_JUSTIFIED`.

Đánh giá full run cục bộ, không gọi model/API:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_llm_adapter.py --dataset .runtime\llm_training\reviewed_v1 --baseline .runtime\llm_training\results\qwen3_run_v1\baseline-predictions.jsonl --candidate .runtime\llm_training\results\qwen3_run_v1\candidate-predictions.jsonl --identities .runtime\llm_training\results\qwen3_run_v1\identities.json --output .runtime\llm_training\results\qwen3_run_v1\evaluation-v1.json
```

Protocol `config/llm-training-v1.json`: tối thiểu100testcases/10connectedgroups; exact task accuracy tăng ít nhất0.05 điểm tuyệt đối; raw schema validity≥0.98; grouped-bootstrap CI phải dương; không giảm anchor/abstention và không critical safety error. Đây là gate đã đăng ký, không được nới để ép pass. Perfect synthetic predictions vẫn INCONCLUSIVE. Grounded synthesis luôn cần blind human review riêng; pipeline hiện không tự promote synthesis.

Registry chỉ chấp nhận manifest đã review quyền/license, human approval và báo cáo PASSED có đúng hash dataset/model/tokenizer/prompt/schema/task/chat-template/weights. Không đổi PENDING thành APPROVED bằng tay để lách thiếu bằng chứng. Hash xác minh tính toàn vẹn/nhận diện, không chứng minh curator trung thực hay nhận định tài chính đúng.

## 5. Tích hợp app và rollback

Notebook mới **không mở server**. Notebook server Qwen2.5 cũ không tự load adapter Qwen3. Interactive reload đã đáp ứng kiểm tra training/export; endpoint app nếu bổ sung sau phải khai báo đúng model revision + adapter identity và trả attestation mà client kiểm tra, đồng thời tuân thủ môi trường hosting.

Giữ base+RAG và toàn bộ validator hiện tại. Chỉ request có đúng task/system/schema mới chọn adapter đã approved; task khác phải dùng base hoặc fallback rõ ràng. Không gán adapter NLU cho Fundamental/Risk synthesis. Khi adapter lỗi/hash mismatch, giữ deterministic fallback; rollback bằng bỏ cấu hình adapter manifest, không xóa V2 hoặc dữ liệu gốc. Không upload secret hoặc raw report riêng vào Git.

Các trạng thái cần phân biệt: `CODE_COMPLETE`, `GPU_SMOKE_PENDING`, `HUMAN_REVIEW_PENDING`, `MODEL_GATE_PENDING/PASSED/REJECTED/INCONCLUSIVE`. Hoàn thành notebook không đồng nghĩa model dự báo tốt hơn.
