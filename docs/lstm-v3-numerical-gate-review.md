# Sửa gate số học và đánh giá lại kết quả LSTM V3

Đã có experiment V4 riêng dựa trên chẩn đoán validation/flat-target: [nghiên cứu và notebook V4](lstm-v4-research-and-training.md). Không thay đổi artifact V3 của báo cáo này.

Ngày 18/10/2026. Không retrain, không thay V2, không sửa `.env`, ZIP gốc hoặc frozen protocol.

## Lỗi và cách sửa

Training cũ tính candidate MAE bằng float64 nhưng có zero-return MAE lấy từ phép cộng/trung bình float32. Ví dụ candidate MAE `0.01739245808927901` và zero MAE `0.017392458394169807` bị đếm là một chiến thắng dù predictions bằng đúng0. Fold0/1 của horizon1 đều bị trường hợp này; calibration chọn alpha0 nên không có tín hiệu LSTM.

- `ml_model_registry.py`: so sánh MAE dùng equivalence tolerance `max(1e-10, 1e-6 * max(abs(candidate), abs(reference)))`. Đây là chống sai số số học, **không phải** ngưỡng lợi ích kinh tế được tối ưu theo test. Mọi seed đăng ký phải có cải thiện vượt tolerance và calibration alpha>0 mới tính fold thắng. Không nới gate để ép pass.
- `ml_validation.py`: candidate/reference/actual của bootstrap đều được chuyển float64 trước khi tính chênh lệch loss.
- `train_lstm_kaggle_v3.py`: giữ tensor training float32, dùng nhãn/evaluation/calibration và baseline MAE float64 thống nhất. Không lấy phép giảm MAE float32 để so với candidate float64 nữa.
- Gate thu thập mọi blocker đã biết thay vì báo chỉ thiếu forward rồi che mất engineering checks/adjusted prices.
- `review_lstm_v3_results.py`: đọc ZIP, kiểm CRC, pinned hashes, model/protocol identities và consistency của predictions; tính lại metric/CI, xuất sidecar mới. Không giải nén hoặc thực thi code trong ZIP, không load Keras/GPU.

Gate code version: `v3-gate-numeric-fix-1`. Cách đánh giá mới được ghi rõ là correction sau khi đã xem kết quả, không giả là một experiment/holdout độc lập mới. Những chỉ tiêu, số fold, seeds, margin Ridge/tree và yêu cầu forward của protocol gốc được giữ nguyên.

## Kết quả ZIP người dùng

Nguồn: `.runtime/ml/v3/lstm-v3-full-20261018-031301-7f4e70d1-results.zip`.

| Horizon | Fold thắng, đủ mọi seed | Yêu cầu | Gate lịch sử sau sửa | Blend |
| --- | --- | --- | --- | --- |
| 1 phiên | 1/3 | Ít nhất2fold | Chưa đạt | 0% |
| 5 phiên | 3/3 | Ít nhất2fold, không thua Ridge/tree quá margin | Đạt bước development | 0% |

Fold0/1 horizon1 có candidate MAE **bằng đúng** zero MAE khi tính cùng precision. Không còn được ghi nhận là thắng. Fold cuối seed42 vẫn có lợi thế MAE một phiên khoảng0,3160% và năm phiên khoảng2,0727% so zero-return, nhưng không thay thế yêu cầu nhiều fold hoặc kiểm chứng độc lập. Đây là giảm **sai số**, không phải lợi nhuận đầu tư.

Cả hai horizon còn thiếu engineering review trong artifact, xác minh giá điều chỉnh và forward holdout mới đủ40phiên. Adjustment và unit gốc vẫn UNKNOWN. Chưa đủ điều kiện promote, kể cả horizon5 đã qua development gate.

Báo cáo cuối cùng:

```text
.runtime/ml/v3/reviews/lstm-v3-full-20261018-031301-numerical-review-v2.json
.runtime/ml/v3/reviews/lstm-v3-full-20261018-031301-numerical-review-v2.md
```

`review-v2` chỉ là phiên bản báo cáo audit, **không phải model V2**. ZIP artifact gốc không thay đổi. Bản review-v1 là lần xuất trước đó; dùng review-v2 có headline gate và bảng selected model cập nhật nhất.

## Cách chạy lại audit, không cần GPU

Chọn tên output mới để giữ báo cáo cũ:

```powershell
.\.venv\Scripts\python.exe scripts\review_lstm_v3_results.py `
  --archive .runtime\ml\v3\lstm-v3-full-20261018-031301-7f4e70d1-results.zip `
  --output .runtime\ml\v3\reviews\new-numerical-review.json
```

Không cần train lại để sửa cách đánh giá của run này. Các prediction CSV không được manifest artifact gốc pin riêng: audit lưu hash ZIP/CSV nhận được và đối chiếu metric với evaluation đã pin, **không xác thực rằng model đã thực sự sinh các dòng đó**. Không suy training-majority direction từ test; chỉ giữ giá trị gốc và ghi giới hạn. Precision trước khi export không thể được phục hồi bằng audit CSV.

Kaggle dataset đã upload chứa snapshot code/runtime cũ. Nếu sau này tạo experiment mới, cần upload script đã sửa và một runtime ZIP mới do builder/refresh tạo; không sửa ZIP hay manifest của run cũ để biến nó thành run mới. Bản sửa local không tự cập nhật Kaggle input. Hiện chưa cần upload hay retrain để đọc báo cáo audit này.

## Kiểm thử

649 test toàn repository qua; có regression float32/float64 tie, alpha0, real improvement không bị loại, bootstrap zero equality, gate nhiều blocker, ZIP/hash/path/seed alignment và input staging Kaggle. Compile và `git diff --check` được kiểm tra riêng. Không coi test fixture là bằng chứng dự báo tốt hơn.
