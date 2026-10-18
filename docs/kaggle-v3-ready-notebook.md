# Chạy LSTM V3 trên Kaggle sau khi bổ sung `.zip.bin`

Notebook mới: `notebooks/train_lstm_kaggle_v3_ready.ipynb`. Giữ nguyên notebook/script V3 cũ và artifact V2. Không cần sửa script training đã upload; notebook này chuẩn bị input đúng dạng trước khi gọi script.

## Các bước chạy

1. Tạo notebook Kaggle, dùng **Import Notebook** để import file `.ipynb` mới. Không paste JSON của file vào code cell.
2. Attach phiên bản mới nhất của dataset V3. Cùng thư mục cần có:
   - `vci_ohlcv_lstm_v3.csv`
   - `dataset.manifest.json`
   - `protocol.json`
   - `train_lstm_kaggle_v3.py`
   - `ml_v3_runtime.aed7aa6849a0.zip.bin` (hoặc ZIP gốc còn nguyên).
3. Bật GPU trong cài đặt notebook.
4. Mặc định `RUN_MODE = 'pilot'`: chạy **Run All** để smoke 3 epochs, 50.000 training endpoints mỗi fold, seed42. Đây không phải full training.
5. Sau khi pilot chạy được, đổi thành `RUN_MODE = 'full'`, rồi chạy lại từ cell cấu hình hoặc **Run All**. Full dùng 3 folds × 2 seeds, tối đa25epochs/trial và early stopping. T4×2 không có nghĩa tự động dùng cả hai GPU.
6. Kết quả ở `/kaggle/working/lstm-v3-<mode>-<timestamp>-<id>/`; cell cuối tạo archive `*-results.zip`. Tải từ Output hoặc link cuối notebook, lưu về ổ D.

Mỗi lần chạy từ cell cấu hình tạo ID/output mới, không ghi đè run cũ. Muốn tiếp tục run bị ngắt: khôi phục toàn bộ run folder dưới `/kaggle/working`, điền `RESUME_RUN_DIR` chính xác và giữ nguyên mode/batch/config. Không chuyển pilot thành full bằng resume.

## Notebook tự xử lý gì?

- Tìm đúng một CSV V3; nếu có nhiều dataset thì yêu cầu điền `DATASET_CSV`, không tùy tiện chọn file đầu.
- Đọc tên ZIP và SHA-256 từ manifest; chấp nhận ZIP nguyên bản hoặc bản `.zip.bin` cùng thư mục.
- Kiểm checksum ZIP, CRC archive và checksum CSV trước khi training. Không tin thư mục Kaggle tự giải nén thay ZIP gốc.
- Tạo staging riêng trong `/kaggle/working`, khôi phục tên ZIP và dùng symlink tới CSV, không copy thêm CSV525MB.
- Nạp script đã attach, kiểm GPU, chạy pilot/full, hiển thị gate và đóng gói output. Không sửa manifest để ép pass, không tự promote model, không cài package lên laptop.

Nếu thiếu `.zip.bin`, cần cập nhật version dataset và input của notebook. Chỉ có thư mục đã giải nén chưa đủ. Sai checksum phải upload lại đúng file/manifest, không bỏ kiểm tra.

Kiểm thử local: 41 tests của notebook input và Phase6 qua; mọi code cell đã compile. Đây là kiểm thử engineering/packaging, **chưa phải chạy Kaggle GPU thực**. Các giới hạn adjustment/unit/forward gate của [model card V3](lstm-v3-model-card.md) không thay đổi.
