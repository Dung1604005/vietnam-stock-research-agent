# Phân tích kỹ thuật và giới hạn

## SMA và EMA

SMA là trung bình cộng giá trong n kỳ. EMA cũng là trung bình động nhưng đặt trọng số lớn hơn cho dữ liệu mới. Cả hai làm mượt nhiễu và mô tả xu hướng trễ; giao cắt không bảo đảm xu hướng sẽ tiếp tục.

## RSI

RSI là bộ dao động động lượng trên thang 0–100. Công thức cơ bản: **RS = mức tăng trung bình / mức giảm trung bình; RSI = 100 − 100/(1+RS)**. Chu kỳ 14 thường được dùng làm mặc định. Mốc 70 và 30 thường được gọi là quá mua và quá bán, nhưng trong xu hướng mạnh RSI có thể ở vùng cao hoặc thấp lâu; không nên coi một mốc RSI là lệnh mua hoặc bán chắc chắn.

## MACD

MACD thường được tính bằng **EMA(12) − EMA(26)**; đường tín hiệu thường là EMA 9 kỳ của MACD. Giao cắt mô tả thay đổi động lượng nhưng dễ tạo tín hiệu nhiễu khi giá đi ngang.

## Bollinger Bands và ATR

Bollinger Bands thường gồm SMA 20 kỳ và hai dải cách trung bình hai độ lệch chuẩn. Dải mở rộng mô tả biến động tăng, không xác định hướng chắc chắn. ATR là trung bình True Range, trong đó True Range là lớn nhất của: high−low, |high−close trước|, |low−close trước|. ATR đo biến động chứ không dự báo chiều giá.

## Volume, hỗ trợ, kháng cự và xu hướng

Volume cho biết lượng giao dịch; nó có thể bổ sung ngữ cảnh cho biến động giá nhưng không chứng minh nguyên nhân. Hỗ trợ và kháng cự là vùng quan sát thấy lực mua hoặc bán trong quá khứ, không phải rào chắn chắc chắn. Xu hướng là hướng biến động quan sát được theo khung thời gian đã chọn; đổi khung thời gian có thể đổi kết luận.

## Giới hạn

Indicator đều được tính từ dữ liệu quá khứ, phụ thuộc tham số và dễ overfit. Nên dùng cùng dữ liệu giá có nguồn và timestamp, kiểm tra nhiều điều kiện và không biến một tín hiệu thành quyết định BUY/SELL tự động.
