# Kỳ báo cáo và kiểm tra độ mới của Research RAG

> Phạm vi báo cáo: số test, latency, corpus và smoke test ghi trong tài liệu là kết quả của lượt triển khai lịch sử, không phải tổng mới ngày 23/10/2026. Trạng thái/code và kiểm chứng mới được tổng hợp tại [project-report.md](project-report.md).

Hệ thống tách ba thời điểm: **kỳ dữ liệu**, **ngày công bố** và **thời điểm tải**. Crawl lại hoặc đăng lại một báo cáo không làm mới kỳ dữ liệu của nó.

## Nhận diện kỳ

`services/reporting_period.py` nhận diện bằng code:

- Quý: `quý 2 năm 2026`, `Q3/2026`, `QIII.2025`.
- Bán niên/6 tháng đầu năm, 9 tháng đầu năm.
- Báo cáo năm và báo cáo thường niên.
- Ngày kết thúc kỳ được ghi rõ trong phần đầu tài liệu bằng tiếng Việt hoặc tiếng Anh.

Metadata gồm năm, quý, loại kỳ, ngày kết thúc, nhãn hiển thị, trạng thái và đoạn bằng chứng dùng để nhận diện. Ngày công bố hoặc năm thành lập công ty không được lấy làm kỳ báo cáo. Kỳ mâu thuẫn hoặc niên độ lệch năm chưa có ngày kết thúc rõ ràng được đánh dấu `AMBIGUOUS`/`UNKNOWN`, thay vì tự đoán.

Nếu tiêu đề và ngày kết thúc kỳ trong nội dung mâu thuẫn, crawler giữ tài liệu ở `QUARANTINED` với lý do `AMBIGUOUS_REPORTING_PERIOD`. Tài liệu cũ hợp lệ vẫn được lưu để phân tích lịch sử.

Khi listing ghi ngày công bố mới hơn tại cùng URL, worker đưa tài liệu vào hàng đợi kiểm tra lại và bỏ HTTP validator cũ. Kỳ của bản PDF trước không được ép lên bản mới. Nội dung thay đổi tiếp tục tạo revision; cùng nội dung chỉ bổ sung kỳ còn thiếu khi có metadata rõ ràng, không tạo document/chunk trùng.

## Lựa chọn theo câu hỏi

- Phân tích một mã hiện tại hoặc hỏi “mới nhất/gần đây”: dùng policy `CURRENT`.
- Nêu rõ năm/quý, so sánh các kỳ hoặc có `as_of`: dùng phạm vi lịch sử, lọc theo kỳ dữ liệu được yêu cầu và ngày công bố đã biết tại thời điểm tra cứu.
- Tra cứu kho bằng CLI mặc định vẫn cho phép tài liệu lịch sử; thêm `--current` để áp policy hiện tại.
- KB lý thuyết không bị áp tuổi báo cáo tài chính này.

Trong `CURRENT`, SQL lọc độ mới **trước khi giới hạn candidate/top-k** ở cả FTS và vector. Với cùng doanh nghiệp và loại tài liệu, kỳ mới hơn đủ điều kiện thay thế kỳ cũ. Báo cáo thường niên và báo cáo tài chính quý vẫn là hai loại khác nhau; hỏi bán niên chỉ chọn bán niên. LLM nhận metadata kỳ cùng mỗi nguồn và phải phân biệt kỳ dữ liệu với ngày công bố.

Mới nhất ở đây là kỳ phù hợp mới nhất **trong tập nguồn đã xác minh mà hệ thống truy cập được**. Không khẳng định đó là báo cáo mới nhất trên toàn Internet.

## Ngưỡng mặc định

Được cấu hình trong `config/research_freshness.json`. Đây là chính sách ứng dụng có thể điều chỉnh, không phải thời hạn công bố theo pháp luật.

| Loại | Tuổi ngày công bố tối đa | Tuổi kỳ dữ liệu tối đa | Bắt buộc xác định kỳ |
|---|---:|---:|---|
| Báo cáo tài chính | 220 ngày | 180 ngày | Có |
| Báo cáo thường niên | 550 ngày | 550 ngày | Có |
| Kết quả kinh doanh | 180 ngày | 240 ngày | Có |
| Investor presentation | 180 ngày | 240 ngày nếu có kỳ | Không |
| Công bố thông tin | 120 ngày | 240 ngày nếu có kỳ | Không |
| Tin tức | 30 ngày | 90 ngày nếu có kỳ | Không |
| Khác | 180 ngày | 240 ngày nếu có kỳ | Không |

Riêng báo cáo tài chính/kết quả kinh doanh nhận diện là **kỳ năm**, tuổi kỳ cho phép là 550 ngày; ngưỡng ngày công bố của loại tài liệu vẫn được áp dụng. Tài liệu chưa xác định được kỳ hoặc có kỳ mâu thuẫn không dùng làm bằng chứng tài chính hiện tại. Kỳ kết thúc trong tương lai so với ngày công bố cũng bị loại.

Tài liệu gắn nhãn `other` nhưng chứa số liệu doanh thu/lợi nhuận cũng được kiểm tra như kết quả kinh doanh; nhãn không rõ không giúp bỏ qua gate xác định kỳ.

Sau khi sửa JSON policy, khởi động lại app/worker để nạp cấu hình mới.

## Khi chỉ có dữ liệu cũ

RAG giữ các tài liệu đó trong phần tham khảo lịch sử, không đưa vào context LLM của câu hỏi hiện tại. Chat gọi Internet research hiện có để tìm thêm nguồn; web evidence đi qua cùng kiểm tra ngày/kỳ/tuổi trước khi tổng hợp.

Research Agent trong luồng phân tích thực tế cũng tìm thêm khi thiếu bằng chứng phù hợp, giới hạn một query, tối đa hai trang và timeout HTTP 8 giây/lần. Kết quả tra cứu được cache 10 phút để tránh gọi lại trong vòng reanalysis; luồng scan batch không mở thêm web search cho từng mã. Giới hạn điều phối không hủy cưỡng bức DNS/HTTP đang chạy và search adapter có cơ chế fallback riêng.

Nếu vẫn thiếu nguồn phù hợp, câu trả lời ghi:

> Chưa có dữ liệu cập nhật đã xác minh đáp ứng yêu cầu. Dữ liệu đến kỳ: Năm 2022 (kết thúc 31/12/2022)…

Đoạn trích chưa rõ ngày công bố hoặc kỳ dữ liệu chỉ được hiển thị với trạng thái tương ứng. Kết quả search/snippet không trở thành bằng chứng tài chính hiện tại chỉ nhờ URL mới hoặc thời điểm tìm kiếm mới.

## Hiển thị và bảo toàn lịch sử

- Citation, bảng tài liệu, nguồn RAG và thư viện báo cáo đều có nhãn **Dữ liệu đến kỳ…**.
- Ngày công bố tiếp tục hiển thị riêng.
- Schema SQLite được migration additive bằng cột JSON kỳ báo cáo và index loại/kỳ.
- Tài liệu đã ingest được backfill metadata khi mở `ResearchStore`; không thay nội dung, checksum, ID hay embedding và không tải lại tài liệu chỉ để sửa metadata.
- Không xóa tài liệu/bản sửa cũ. Các truy vấn lịch sử vẫn chọn phiên bản theo ngày công bố/as-of đã có.

Chạy kiểm tra local, không gọi LLM/web:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\search_research.py "báo cáo tài chính FPT mới nhất" --ticker FPT --current
```

Tìm lịch sử trên app: `Tìm báo cáo tài chính FPT quý 2 năm 2022` hoặc `So sánh quý 1 và quý 2 năm 2022 của FPT`. Đây là truy vấn tài liệu; không thay thế backtest/giá point-in-time.

Kiểm thử tập trung:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_research_freshness.py
```

Các trường hợp kiểm thử gồm báo cáo cũ đăng lại gần đây, nhiều kỳ trước top-k, tuổi riêng theo loại tài liệu, metadata thiếu/mâu thuẫn, so sánh quý cũ, bán niên, backfill không embed lại, không gọi LLM với evidence cũ và web fallback/cache.

Kiểm chứng ngày 17/10/2026: **394 tests passed** trên toàn repository; smoke-test màn hình thư viện có **0 exception**. Kiểm tra kho thực tế bằng lệnh `--current` trả thông báo thiếu dữ liệu cập nhật và giữ tài liệu năm 2022 trong phạm vi lịch sử. Web fallback được kiểm thử bằng adapter giả lập, không phải chứng nhận mọi nguồn Internet luôn truy cập được.
