# LSTM V4 — điều tra V3 và ưu tiên dự báo phiên kế tiếp

Ngày18/10/2026. Trạng thái cập nhật: **USER_GPU_RUN_REVIEWED / DEVELOPMENT_GATE_PASS / FORWARD_VALIDATION_PENDING**. ZIP `lstm-v4-full-20261018-040730-3c392f60-results.zip` đã kiểm 31 checksum, tính lại MAE khớp; cả hai horizon thắng 2/3 fold. Đây vẫn là development, không phải bằng chứng hiệu quả đầu tư. Theo yêu cầu người dùng, UI bổ sung lựa chọn V2/V4 với **experimental blend 20%**, không sửa eligibility/gate gốc; xem [cách chọn model](lstm-ui-model-selection.md). Không cài package/model lên C và không có broker/đặt lệnh.

## 1. Vì sao V3 chưa đạt một phiên?

Đã đọc đúng ZIP người dùng `lstm-v3-full-20261018-031301-7f4e70d1-results.zip`, kiểm hash và đối chiếu dataset đã train. Audit riêng được lưu:

```text
.runtime/ml/v4/v3-failure-audit-v1.json
```

### Những điều đã xác minh

| Vấn đề | Bằng chứng trong run V3 | Hệ quả có thể khẳng định |
| --- | --- | --- |
| Validation quá mỏng ở fold đầu | 2.728 dòng nhưng chỉ **5 ngày** đủ endpoint, dù khoảng lịch03–06/2023; best checkpoint cả2seed ở epoch1 | Không thể coi hàng nghìn mã/ngày là hàng nghìn ngày độc lập; coverage để chọn model rất yếu |
| Median lợi nhuận một phiên bằng0 | Median=0 ở cả12 block train/validation/calibration/test; test flat25,36% /24,87% /22,63% | Zero-return là đối thủ phù hợp và khó thắng bằng MAE; không nên coi model dự báo0 là lỗi kỹ thuật tự thân |
| Calibration bỏ tín hiệu một phiên | Alpha0 ở cả2seed của fold0/1, predictions sau calibration bằng0 | Chưa có cải thiện LSTM trên các fold đó; **không chứng minh raw model trước calibration đã bằng0** |
| Checkpoint không ưu tiên metric cần dùng | Model V3 dùng chung Dense2, MAE chuẩn hóa1/5phiên; monitor `val_loss` gồm regularization | Checkpoint tối ưu loss gộp, không nhất thiết tốt nhất về MAE raw-return một phiên |
| Số học làm sai gate cũ | Float32 zero MAE so với float64 candidate tạo chiến thắng giả | Đã sửa riêng; không được dùng trạng thái chỉ chờ forward cũ làm bằng chứng chất lượng |

Về lý thuyết, MAE gắn với ước lượng median, nên việc dự báo gần0 trên phân phối có median0 không bất thường. Đây **không** phải chứng minh conditional median luôn0 hoặc feature không thể có tín hiệu. [Nguồn scikit-learn chính thức về quantile0,5/MAE](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html).

### Giả thuyết cần kiểm chứng, không phải kết luận nguyên nhân chắc chắn

Hai horizon cạnh tranh gradient, độ co về median, nhóm thanh khoản thấp/đứng giá và độ nhiễu cao của một phiên có thể làm suy yếu tín hiệu. Train lâu hơn/tăng layer không được chứng minh là giải pháp. OHLCV cũng không chứa toàn bộ tin tức, báo cáo hay thông tin thị trường. Không có cơ sở gọi mọi biến động lớn là dữ liệu lỗi hoặc xóa chúng khỏi test.

## 2. V4 thay đổi những gì?

1. **Chia fold theo ngày thực sự có endpoint**, nhưng gap vẫn tính trên lịch phiên nguồn đầy đủ. Purge nhãn5phiên ở mọi biên, không bỏ qua khoảng trống. Split không dựa vào độ lớn/dấu của lợi nhuận tương lai. Yêu cầu tối thiểu train120, validation30, calibration30, test40 ngày đủ mẫu; thiếu thì dừng, không âm thầm train với5ngày.
2. **25 feature causal**: giữ17feature V3, thêm flat-ratio5/20, volume CV20, turnover-ratio20, thân nến theo previous close, tỷ lệ râu trên/dưới và return-mean5. Tính được từ một mã ở runtime; không thêm proxy cross-section. Có kiểm causality/cache parity. CSV V3 đã hash được tái dùng17feature để tránh tính lại tốn CPU.
3. **LSTM32 + nhánh quan sát cuối** và head riêng `return_1`, `return_5`, `direction_1` (DOWN/FLAT/UP). Direction là tác vụ phụ học feature; không tự được coi là xác suất đầu tư đã calibrated. Dùng lớp Keras chuẩn để save/reload không cần custom/Lambda object. [Keras multi-output training](https://keras.io/guides/training_with_built_in_methods/).
4. **Ưu tiên một phiên**: Huber trên return đã chia volatility biết tại origin; loss weights1,0 /0,35 /0,15 cho return1/return5/direction. Target training clip cố định±8 đơn vị chuẩn hóa; **không clip/drop nhãn evaluation**. Tensor train float32, mọi metric/baseline/CI float64. Giảm L2 xuống1e-5 và dropout ngoài LSTM0,1 là giả thuyết thiết kế, chưa phải tác động được chứng minh.
5. **Checkpoint theo `val_raw_return_1_mae`**, tính trên toàn bộ validation hợp lệ bằng nhãn gốc; không dùng `val_loss` gộp hoặc test để chọn. Seed cuối cùng cũng chỉ chọn theo validation của fold cuối. Resume giữ best-score cũ, không cho checkpoint kém hơn ghi đè bản tốt đã có.
6. **Calibration có nhận diện flat**, alpha gồm0..1 và threshold cố định None/0,5/0,65/0,8; chỉ fit trên block calibration-fit. Alpha0 vẫn được phép nếu model không có tín hiệu; không loại lựa chọn đó để ép pass. Residual interval fit ở phần calibration khác đã purge, không dùng chung phần chọn alpha. Khoảng này vẫn không bảo đảm coverage tương lai.
7. **Baseline/gate không bị nới**: zero/persistence/train mean/median/Ridge/tree cùng endpoint và nhãn; Ridge/tree dùng25feature, fit train và chọn bằng validation. CI theo block ngày, metric flat/nonflat/activity và phân loại3lớp được tách riêng. Gate mỗi horizon vẫn cần≥2/3fold, đủseeds, không thua Ridge/tree quá1%, forward mới40phiên cùng các kiểm chứng dữ liệu/engineering.
8. **Artifact truy vết tốt hơn**: khóa CSV/protocol/trainer/runtime/libs; manifest pin cả trial predictions/checkpoints, source code và runtime ZIP thay vì chỉ model cuối. V4 artifact schema riêng, không giả tương thích adapter V3; mặc định luôn experimental/shadow0% và không tự bật trong app.

LSTM dùng `recurrent_dropout=0` cùng activation mặc định và dropout ngoài để phù hợp điều kiện đường GPU nhanh; không bảo đảm tốc độ trước đo pilot. [Keras LSTM GPU requirements](https://keras.io/api/layers/recurrent_layers/lstm/).

## 3. Preflight trên dataset thực đã có

Không sinh lại/sửa CSV, giữ1triệu dòng và991mã. Cả333.690endpoint gốc vẫn được giữ;25feature được tính khi chạy notebook.

| Fold | Ngày train đủ mẫu | Validation | Calibration | Test |
| --- | --- | --- | --- | --- |
| 0 | 423 | 33 | 34 | 77 |
| 1 | 597 | 33 | 34 | 77 |
| 2 | 771 | 33 | 34 | 79 |

Sau purge, mỗi calibration được tách thành17ngày fit alpha/threshold và12ngày fit interval. Đây vẫn là ít ngày độc lập so với số dòng; không lấy số mã để phóng đại cỡ mẫu. Báo cáo `.runtime/ml/v4/data-preflight-v1.json` có sample counts và từng biên ngày. Adjustment/unit tiếp tục UNKNOWN.

V4 đổi splits nên **không so headline MAE với báo cáo V3 một cách trực tiếp**. V3 đã xem nhiều lần; toàn bộ lịch sử này là development. Cần chạy hai model trên cùng mã/ngày/horizon hoặc forward observations mới để kết luận cải thiện thật.

## 4. Cách chạy ngay trên Kaggle

Notebook: **`notebooks/train_lstm_kaggle_v4_ready.ipynb`**. Code nguồn: `notebooks/train_lstm_kaggle_v4.py`. Protocol: `evaluation/configs/lstm-v4-protocol.json`.

1. Import notebook mới bằng Import Notebook, không paste JSON vào Python cell.
2. Attach **dataset DataVCI_V3 hiện có**. Chỉ cần CSV và `dataset.manifest.json` gốc cùng thư mục. **Không upload lại CSV, không cần ZIP/.bin/script V4 trong dataset**: code/runtime/protocol đã được nhúng trong notebook.
3. Bật GPU. Nếu session đang giữ code V3 thì restart session trước khi Run All.
4. Giữ `RUN_MODE = 'pilot'`, chạy Run All:3epochs/seed42, tối đa50.000training endpoints mỗi fold; validation/calibration/test không bị subsample để làm đẹp kết quả.
5. Xem pilot có chạy được, memory/time và các ngày/samples đủ chuẩn. Muốn train đầy đủ, đổi `RUN_MODE = 'full'` rồi Run All từ đầu:3folds×2seeds, max35epochs/trial, early stop theo raw validation MAE1phiên. Mỗi run có folder/ID mới, không đè bản cũ.
6. Tải `lstm-v4-...-results.zip` ở Output/cell cuối. Lưu vào `.runtime/ml/v4/` trên D; không ghi đè V2/V3.

`VARIANT='regression_only'` là ablation tùy chọn (direction loss0); chạy output riêng, đăng ký trước cách so sánh, không xem test rồi chọn variant thắng và gọi đó là independent result. Không cần chạy nhiều architecture để demo.

Mất phiên: chỉ resume khi còn/đã khôi phục toàn bộ run directory cùng checkpoints và config; điền `RESUME_DIR`, giữ nguyên mode/variant/code/data. Không resume pilot thành full hoặc sửa protocol/loss để tiếp tục run cũ. Callback patience có thể khởi động lại, nhưng checkpoint chỉ được thay nếu vượt best raw validation score đã lưu. Notebook không tránh quota, không dùng keepalive và không tự chuyển dịch vụ trả phí.

Notebook có thể chạy Colab nếu đặt CSV+manifest dưới `/content/stock-agent-v3`; không tự mount Drive hoặc upload dữ liệu. Không hứa thời gian train trước khi đo pilot trên GPU thực. T4×2 không tự sử dụng cả2GPU.

## 5. Kiểm chứng đã thực hiện và còn thiếu

Đã kiểm feature causality, cache parity, fold gaps/purge/coverage, class weights, alpha0, probabilities, shared runtime import và code cells. Bộ regression toàn dự án có666test qua. TensorFlow trong venv D đã chạy một batch3head, rồi smoke end-to-end **chỉ với1.600dòng synthetic CPU**:3folds, checkpoint/reload, calibration và export thành công. Kết quả synthetic không dùng vào gate chất lượng hoặc dữ liệu app.

Bản ghi trước khi nhận kết quả: chưa train V4 GPU trên dữ liệu thị trường. Hiện người dùng đã cung cấp full run nêu đầu tài liệu; đã xác nhận development gate nhưng chưa forward validation. Giữ V2/V3 để so sánh/rollback. Không nới gate hoặc thay nhãn để bắt model pass. Adapter UI mới đã nạp chính model V4 và chạy inference + blend 80/20 trên fixture synthetic để kiểm kỹ thuật; đó không phải kiểm chứng chất lượng trên thị trường.

CLI diagnostic local (không training):

```powershell
.\.venv\Scripts\python.exe scripts\audit_lstm_v3_failure.py --archive .runtime\ml\v3\lstm-v3-full-20261018-031301-7f4e70d1-results.zip --data .runtime\ml\v3\dataset-1\vci_ohlcv_lstm_v3.csv --output .runtime\ml\v4\new-v3-audit.json
.\.venv\Scripts\python.exe scripts\preflight_lstm_v4.py --data .runtime\ml\v3\dataset-1\vci_ohlcv_lstm_v3.csv --output .runtime\ml\v4\new-data-preflight.json
```

Chọn tên output mới, không overwrite audit cũ. `.env`, raw/V2/V3 và ZIP kết quả gốc không bị chỉnh sửa trong đợt nâng cấp này.
