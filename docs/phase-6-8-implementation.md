# Triển khai Phase 6–8 và cách chạy

> Cập nhật phạm vi 23/10/2026: các ô “Train GPU” bên dưới là handoff ở mốc triển khai, đã có review artifact LSTM V3/V4 sau đó. Xem [V3 card](lstm-v3-model-card.md), [V4 review](lstm-v4-research-and-training.md), [V2/V4 UI](lstm-ui-model-selection.md). Không đồng nhất train hoàn tất với promotion/forward gate.

> Phạm vi báo cáo: số test, latency, corpus và smoke test ghi trong tài liệu là kết quả của lượt triển khai lịch sử, không phải tổng mới ngày 23/10/2026. Trạng thái/code và kiểm chứng mới được tổng hợp tại [project-report.md](project-report.md).

Cập nhật 18/10/2026. Phạm vi: engineering, notebook training và demo; không broker/đặt lệnh, không tuyên bố chất lượng đầu tư chỉ từ test phần mềm. Code mới giữ V2, không sửa `.env`, không tự tải model, upload dữ liệu hay cài training dependencies lên laptop.

## 1. Trạng thái bàn giao

| Phase | Đã thực hiện và kiểm chứng local | Còn cần thực hiện ngoài phiên code |
| --- | --- | --- |
| 6 — ML/LSTM | Dataset V3 mới; 17 feature train/runtime dùng chung; fold tiến thời gian/purge; baseline CPU thật; notebook LSTM batched; registry/checksum; shadow/toggle, fail-closed weight | Train GPU; review engineering report; xác minh giá điều chỉnh/đơn vị; forward holdout mới trước khi promote |
| 7 — Qwen3 fine-tune | Notebook Colab load 4-bit, QLoRA pilot/full, checkpoint/resume, raw A/B, export/reload; dataset rights/privacy/group split; registry/task-contract và adapter identity | Human-reviewed dataset đúng task app; GPU smoke/full run; A/B đủ mẫu; human approval; endpoint tương thích nếu muốn dùng adapter live |
| 8 — Demo/runtime | STANDARD/QUICK/DEEP/OFFLINE; tiến độ theo node thật; replay OHLCV có checksum; local health; nguồn/ngày theo nhóm dữ liệu; MD/HTML/audit JSON có sanitizer | Capture replay thật được phép sử dụng; rehearsal trên máy người dùng; đo latency/Colab live |

`CODE_IMPLEMENTED` và kiểm thử offline **khác** `MODEL_GATE_PASSED`. Hiện chưa có V3 LSTM hoặc adapter Qwen3 được promote. Không cần chờ model mới để dùng các cải tiến demo/runtime.

## 2. Dataset LSTM V3 đã tạo ở đâu?

Đường dẫn đầy đủ:

```text
D:\Download\thayque\Agent\.runtime\ml\v3\dataset-1\vci_ohlcv_lstm_v3.csv
```

Thống kê thực tế từ `dataset.manifest.json`:

| Thuộc tính | Giá trị |
| --- | --- |
| Kích thước CSV | 524.950.632 byte, khoảng 500,6 MiB |
| Dòng raw được giữ | 1.000.000 |
| Mã | 991 |
| Feature V3a | 17, single-ticker, tính từ quá khứ |
| Window | 30 quan sát nguồn liên tục trong segment |
| Endpoint đủ điều kiện | 333.690 |
| Segment | 8.255 |
| Structural quarantine | 0; không đồng nghĩa toàn bộ dữ liệu tài chính đã chính xác |
| Price adjustment / unit | UNKNOWN; không tự sửa thành ADJUSTED hoặc VND |

Endpoint ít hơn số dòng vì cần warm-up, lịch sử liên tục, thanh khoản/context và đủ nhãn cả 1/5 phiên. Không xóa dòng vì biết lợi nhuận tương lai lớn. Giá điều chỉnh chưa rõ và universe hiện tại không phải point-in-time vẫn là giới hạn thật; dataset này chưa đủ để khẳng định hiệu quả đầu tư hoặc cho V3 blend.

Các file cùng thư mục cần đưa lên notebook cùng CSV:

- `dataset.manifest.json` và `protocol.json`;
- **`ml_v3_runtime.aed7aa6849a0.zip`**, chính là runtime ZIP manifest đang chỉ định;
- script `notebooks/train_lstm_kaggle_v3.py` từ repository.

ZIP `ml_v3_runtime.zip` cũ được giữ để đối chiếu, không chọn thay ZIP mới. Nếu code shared được thay đổi trước lần training sau, có lệnh refresh rõ ràng ở [model card](lstm-v3-model-card.md); không tự sửa checksum để bỏ kiểm tra.

Baseline CPU trên dữ liệu thật hiện có đã chạy đủ 3 folds × 2 horizons, gồm zero/persistence/train mean/median/Ridge/tree; tree thực sự chạy thành công. Báo cáo:

```text
.runtime/ml/v3/baselines-2.json
```

Đây là **development walk-forward**, không phải kết quả LSTM chưa train hoặc holdout độc lập. File raw, cleaner cũ, notebook V2 và bốn file artifact V2 vẫn nguyên; checksum V2 đã đối chiếu đúng manifest.

## 3. Train LSTM V3 trên Kaggle hoặc Colab

**Kaggle tự giải nén ZIP:** dùng bản mới [train_lstm_kaggle_v3_ready.ipynb](../notebooks/train_lstm_kaggle_v3_ready.ipynb), nhận `.zip.bin` và staging/checksum tự động. Xem [cách chạy Run All](kaggle-v3-ready-notebook.md). Bản cũ bên dưới vẫn được giữ nguyên.

1. Chỉ upload dữ liệu nếu quyền nguồn cho phép; không public dataset mặc định.
2. Import `notebooks/train_lstm_kaggle_v3.ipynb` bằng chức năng upload/import notebook. **Không paste nội dung JSON `.ipynb` vào Python cell.**
3. Kaggle: attach gói dữ liệu mới và script `.py` ở trên; chúng nằm dưới `/kaggle/input`. Colab: đặt CSV/manifest/protocol/runtime ZIP/script vào `/content/stock-agent-v3`, hoặc sửa `INPUT_ROOT` trong notebook.
4. Chọn GPU và chạy cell nạp script, rồi pilot: 3 epochs, 50.000 training endpoints mỗi fold, seed42. Đây là smoke tài nguyên, không phải model đủ điều kiện promote.
5. Sau khi xem pilot, bỏ comment cell full: 3 folds × seeds42/123, tối đa25epochs/trial với early stopping. Dùng output folder mới; không lấy pilot folder làm full run.
6. Mất phiên: giữ checkpoint và toàn bộ run folder, chạy lại đúng arguments + `--resume`. Không thay config/dataset để tiếp tục một run cũ.
7. Tải toàn bộ output về D, ví dụ `.runtime/ml/v3/artifacts/run-1/`. Giữ đủ `model.keras`, `spec.json`, `evaluation.json`, `protocol.json`, `manifest.json`.

Không có thời gian GPU đã đo để hứa “train bao lâu”. Full là nhiều trial, phải đo pilot trên GPU được cấp trước. Không tự dùng cả hai GPU hoặc cấp dịch vụ trả phí.

Default vẫn V2. Muốn xem V3 shadow sau khi đã có artifact, người dùng có thể đặt `STOCK_AGENT_LSTM_V3_SHADOW_DIR` tới run folder; shadow luôn **0%** trọng số. Chọn V3 riêng dùng `STOCK_AGENT_FORECAST_MODEL_VERSION=v3` và `STOCK_AGENT_LSTM_V3_ARTIFACT_DIR`; không có thư mục hợp lệ thì fallback baseline có nhãn, không giả đã chạy V3. Chưa qua gate → không blend; tối đa20% chỉ khi đủ điều kiện và input tương thích.

Protocol cần cả fold evidence, engineering review, dữ liệu điều chỉnh phù hợp và **40 phiên forward mới**. Ngày chờ nhãn không thể thay bằng GPU nhanh. Xem [LSTM V3 model card](lstm-v3-model-card.md) về identity ổn định, forward ledger và rollback.

## 4. Chạy fine-tune Qwen3-4B trên Colab T4

Notebook chính: **`notebooks/finetune_qwen3_4b_lora_v1.ipynb`**. Bản Python tương ứng là `notebooks/finetune_qwen3_4b_lora_v1.py`.

1. Colab → **File → Upload notebook**, chọn `.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Chạy cell cài dependency **trong Colab**. Nếu yêu cầu restart runtime thì restart rồi chạy tiếp từ cell định nghĩa. Không chạy pip của notebook trong `.venv` Windows.
4. Chạy CUDA readiness; notebook kiểm GPU/library trước khi download model.
5. Đọc quyền dữ liệu và license. Chỉ sau khi đã kiểm tra, đổi `UPLOAD_RIGHTS_CONFIRMED=True` và `LICENSE_REVIEWED=True`.
6. Chọn một trong hai nguồn dữ liệu:
   - Muốn thử ngay pipeline: `ALLOW_SYNTHETIC_DEMO=True`. Notebook tự tạo 288 ví dụ tự biên soạn, gắn nhãn nháp/demo; không đọc chat riêng hoặc `.env`.
   - Muốn adapter có thể xét tích hợp: giữ False, upload `notebook-upload.zip` từ dữ liệu đã con người review, đúng system prompt/schema/task app. Cách tạo ở [data policy](llm-finetuning-data-policy.md).
7. Giữ `TRAIN_FULL=False`, `MAX_SEQUENCE_LENGTH=512`, output mới `/content/qwen3_lora_pilot_v1`, chạy pilot50steps. Pilot chỉ đánh giá validation; không dùng test để chỉnh model.
8. Xem GPU memory, thời gian, output và lỗi. Full run: đổi `TRAIN_FULL=True`, dùng output mới `/content/qwen3_lora_full_v1`, khóa config rồi chạy2epochs.
9. Cell cuối tải `*-results.zip`; lưu trên D dưới `.runtime/llm_training/results/`. Archive giữ checkpoint, adapter, dataset lock, baseline/candidate predictions và identities.
10. Cell `run_saved_inference(...)` reload trọng số đã lưu để thử prompt riêng ngay trong notebook. Nhãn vẫn `EXPERIMENTAL_NOT_APPROVED`.

Base là `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; NF4 4-bit/FP16, LoRA rank8, microbatch1/accumulation8, answer-only loss. Dependency pins là cấu hình ứng viên, **chưa có GPU smoke trong phiên code này**.

Resume: dùng `RESUME='auto'` với nguyên dataset/config/output của run bị ngắt. Checkpoint local Colab mất nếu runtime bị xóa; muốn lưu Drive phải tự mount và có quyền upload rõ ràng. Không có tự động upload hoặc chống ngắt phiên bằng keepalive.

### Vì sao train xong chưa tự bật trong app?

Loss giảm không đủ. Pilot nháp cố ý dùng schema đơn giản, **không phải schema NLU của app**; tuyệt đối không gán nó cho Manager/Fundamental/Risk. Để tích hợp cần dữ liệu đúng task, raw A/B trên grouped holdout đủ mẫu, kiểm privacy/safety và manifest review. Registry kiểm đúng base/tokenizer/prompt/schema/chat-template/dataset/weights; task khác không được dùng nhầm adapter.

Notebook này là training và interactive inference, **không mở server/tunnel**. Endpoint Qwen2.5 đang cấu hình trước đó không tự trở thành Qwen3 adapter server. Client mới hỗ trợ identity handshake cho một endpoint tương thích sau này; vẫn cần triển khai endpoint được môi trường hosting cho phép. `.env` của bạn không bị thay; chỉ đổi tên model trong app không có nghĩa đã thay model trên server.

[Hướng dẫn chi tiết, lệnh đánh giá và quyền dữ liệu](llm-finetuning-data-policy.md) · [Adapter model card](llm-adapter-model-card.md).

## 5. Dùng các cải tiến demo ngay bây giờ

Chạy từ thư mục repository thực tế:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Sidebar có:

- STANDARD: giữ cấu hình hiện hành theo intent.
- QUICK: 0 LLM, tối đa30giây/80tool; vẫn có thể lấy market data thật.
- DEEP: các agent chia sẻ trần tối đa4LLM/180giây/250tool, không phải mỗi agent một quota.
- OFFLINE: 0 outbound/LLM, dùng KB/replay đã lưu; mã thiếu báo thiếu, không gọi mạng để bù.

Budget profile chỉ siết intent budget; không cam kết ngắt mọi hàm Python đang block đúng giây. Không có selector mở rộng6calls trong UI đợt này.

Progress hiển thị node/attempt thực, không in trước mọi agent là Done. Phân biệt nguồn LIVE_FETCH/CACHE/SNAPSHOT_REPLAY/MODEL_ESTIMATE và giữ ngày gốc. Có local health không gọi mạng/GPU; file model hiện diện không đồng nghĩa đã verify hoặc generate thành công. Báo cáo tải được MD/HTML/audit JSON có cùng lớp lọc dữ liệu nhạy cảm.

Replay thật cần capture rõ ràng bằng `scripts/capture_demo_bundle.py` khi có Internet/quyền dùng dữ liệu. Đợt này chỉ chạy replay fixture trong test, **không tạo fixture giả làm dữ liệu thật cho buổi demo**. Các bước capture và kịch bản trình bày: [demo-runbook.md](demo-runbook.md).

## 6. Bằng chứng kiểm thử của lượt triển khai

- Toàn bộ suite: **620 passed**, 23,35giây trên môi trường local của phiên này.
- Có AppTest UI thực: Offline → worker → response → chart/downloads → rerun, không lặp request; network bị patch chặn.
- QUICK không gọi LLM; OFFLINE không fallback mạng; sai checksum bundle/adapter bị từ chối; V3 không đủ gate không nhận20% legacy weight.
- Regression synthetic `evaluate_upgrade.py --strict`: toàn bộ gate được đo đều pass; recall@5=0,9625, MRR=0,9375. Báo cáo `.runtime/evaluations/phase6-8-regression-20261018.json`. Không coi đây là live citation precision hoặc chất lượng đầu tư.
- Compile Python modules/scripts/notebooks và code cells notebook, đồng bộ Python/notebook; `git diff --check` sạch.
- Bốn checksum artifact V2 đều trùng manifest; không đổi default V2 hoặc publish dataset/model.

Lệnh tái kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\evaluate_upgrade.py --strict
```

Chưa được đo: độ chính xác fine-tuned model/V3, GPU throughput/peak RAM thực, independent forward outcomes, chất lượng LLM/citation live và P95 của Colab người dùng. Đây là các bước nghiệm thu tiếp theo, không phải checkbox được thay bằng synthetic tests.
