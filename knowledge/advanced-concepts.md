# Khái niệm đầu tư và kiểm định nâng cao

## Beta, alpha, CAPM và Sharpe

Beta đo mức đồng biến của lợi suất tài sản với thị trường theo một mẫu dữ liệu và benchmark đã chọn. **CAPM: E(R_i) = R_f + β_i[E(R_m) − R_f]**. Alpha thường là phần lợi suất vượt mức mô hình hoặc benchmark sau điều chỉnh rủi ro. **Sharpe = (R_p − R_f) / σ_p**, so sánh lợi suất vượt lãi suất phi rủi ro với biến động tổng. Các chỉ số thay đổi theo cửa sổ dữ liệu, tần suất và benchmark.

## Efficient Market Hypothesis và factor investing

Giả thuyết thị trường hiệu quả mô tả mức độ giá phản ánh thông tin; các dạng yếu, bán mạnh và mạnh dùng tập thông tin khác nhau. Đây là giả thuyết để kiểm định, không phải khẳng định mọi giá luôn đúng. Factor investing xây danh mục theo các đặc tính có hệ thống như value, size, momentum hoặc quality; phần bù lịch sử có thể suy giảm và không bảo đảm tương lai.

## Backtesting

Backtest áp dụng quy tắc vào dữ liệu lịch sử để ước lượng hành vi rủi ro–lợi suất. Thiết kế tốt cần phân tách theo thời gian, chi phí giao dịch, dữ liệu point-in-time và đánh giá ngoài mẫu. Walk-forward gần quá trình thực tế hơn việc fit và đánh giá trên cùng một giai đoạn.

## Look-ahead bias, survivorship bias và data leakage

Look-ahead bias xảy ra khi mô hình dùng thông tin chưa thể biết tại thời điểm quyết định. Survivorship bias xảy ra khi chỉ giữ các mã còn tồn tại và loại doanh nghiệp hủy niêm yết hoặc thất bại. Data leakage là mọi đường truyền khiến thông tin target hoặc tương lai lọt vào feature, preprocessing hay lựa chọn mô hình. Cả ba có thể làm kết quả lịch sử đẹp giả tạo.

## Giới hạn

Backtest chỉ là bằng chứng lịch sử dưới các giả định. Structural break, fat tails, lựa chọn chiến lược sau nhiều lần thử và dữ liệu chưa điều chỉnh đều có thể làm hiệu quả ngoài mẫu kém xa báo cáo.
