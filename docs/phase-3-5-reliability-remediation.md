# Kế hoạch khắc phục các hạn chế Phase 3–5

Phạm vi: nâng cấp demo nghiên cứu, không giao dịch tự động; không thêm dịch vụ trả phí. Colab hiện có phục vụ LLM. Không coi kiểm thử phần mềm là bằng chứng mô hình dự báo có lợi nhuận.

## Thứ tự và tiêu chí nghiệm thu

1. **Chuẩn hóa tài chính:** giữ nguyên lịch sử, đơn vị/scale, tiền tệ, kỳ, annual/quarter/YTD, hợp nhất và assurance; chỉ so sánh/derive khi tương thích. Fundamental và chart dùng cùng lớp facts; thiếu metadata phải hiển thị hạn chế.
2. **Grounding từng nhận định:** claim gắn quote, nguồn, mã, metric, kỳ, đơn vị và giá trị khi có. Chặn sai mã/kỳ/số; phân biệt kiểm chứng anchor với tính đúng của lập luận. Review cũ không được gắn nhãn đã xác minh từng claim.
3. **Vòng bổ sung bằng chứng:** LLM chỉ đề xuất một trong các task được cho phép; Manager kiểm tra quyền nguồn, ngân sách, số vòng và trùng lặp. Code gọi Data/Research, tính lại phân tích chịu ảnh hưởng; dừng khi không có bằng chứng mới. Không cho model tự gọi URL/code tùy ý hoặc sửa score.
4. **Risk theo coverage/ngành:** tách rủi ro quan sát được khỏi thiếu dữ liệu; UNKNOWN/PARTIAL/ASSESSED; liệt kê phần thiếu và đóng góp điểm. Không suy ra rủi ro thấp chỉ vì thiếu báo cáo. Không giả định đơn vị giá.
5. **Chart và giải thích:** nhiều chart, lịch sử tài chính, phân biệt annual/quarter/YTD; không vẽ growth từ kỳ không tương thích. Giải thích điểm bằng input/source/period và đóng góp score.
6. **KG có bằng chứng:** entity/fact/event/claim liên kết nguồn và thời gian; quan hệ kinh tế cần review rõ ràng, không biến đồng xuất hiện thành quan hệ nhân quả. Re-ingest idempotent; graph-RAG vẫn opt-in trước khi đo lợi ích.
7. **Đánh giá thực tế:** kiểm thử hồi quy và ca sai issuer/kỳ/đơn vị; công cụ xuất gói review từ snapshot thật để người dùng đánh giá. Chưa có đánh giá độc lập phải ghi PENDING, không thay bằng dữ liệu giả rồi tuyên bố chất lượng đầu tư.

## Phân chia công việc

- Nhánh financial/charts: metadata, canonical facts, tính so sánh và chart.
- Nhánh claim grounding: schema claim/proposal, kiểm tra evidence và gateway.
- Nhánh risk/graph: coverage ngành, score attribution, quan hệ graph có nguồn.
- Nhánh tích hợp: Fundamental, Manager follow-up, phản hồi/UI, bộ đánh giá và kiểm thử toàn hệ thống.

## Giới hạn phải giữ

- PDF ảnh cần OCR; metadata không rõ thì không tự đoán là đã kiểm toán/hợp nhất/giá điều chỉnh.
- Không dùng lại test holdout để chọn tham số rồi gọi đó là độc lập. LSTM V2 giữ trạng thái nghiên cứu; không train hoặc hứa cải thiện lợi nhuận trong đợt này.
- Không bảo đảm nguồn free luôn có báo cáo mới hoặc truy cập từ cloud; phải thể hiện thiếu dữ liệu.
- Review anchor không chứng minh quan hệ nhân quả hoặc khuyến nghị đầu tư đúng.

## Trạng thái

Đã tích hợp phần code của cả 7 nhóm. Nghiệm thu chất lượng thực tế chưa hoàn tất: cần endpoint Colab đang chạy, dữ liệu có đủ metadata và người review độc lập. Không đánh dấu các gate này hoàn thành chỉ từ fixture.

| Nhóm | Kết quả triển khai | Điều kiện còn cần xác minh ngoài kiểm thử |
| --- | --- | --- |
| Tài chính | Canonical facts/history, scale/basis/consolidation/assurance, YoY/cash conversion có lineage; provider giữ nhiều kỳ và không đoán ratio scale | Nguồn phải thực sự công bố đơn vị/cơ sở kỳ; PDF bảng phức tạp vẫn có thể bị từ chối |
| Grounding | Tối đa 2 claim ngắn/review, kiểm tra nguồn/quote/issuer/metric/period/unit; legacy tách nhãn | Anchor đúng chưa chứng minh suy luận đúng; cần review tiếng Việt từ Colab thật |
| Follow-up | Task allowlist, Manager duyệt, tối đa 1 action, fingerprint chống vòng không tiến triển, recompute và vô hiệu review cũ | Call/tool/deadline còn đủ; provider có thể không trả được nguồn mới |
| Risk | PARTIAL/UNKNOWN/ASSESSED, checklist ngành, coverage/missing, đóng góp điểm, không giả đơn vị giá | Thiếu NPL/CAR/audit/covenants vẫn phải hiện thiếu; không tự nhận đã đo toàn bộ rủi ro |
| Chart/explain | Nhiều chart; điều khiển annual/quarter và 2Y/3Y/5Y/8Q/ALL; bảng chỉ số và đóng góp điểm | Không tự dựng lại lịch sử không có trong nguồn; không suy luận nguyên nhân tăng trưởng chỉ từ chart |
| KG | Company-event-period, metric-period, reviewed Claim→Evidence; quan hệ risk cần review, idempotent | Chưa có independent relational holdout; Graph-RAG giữ opt-in, không tự bật |
| Evaluation | Xuất snapshot thật thành gói review đóng băng, rubric và tổng hợp chỉ từ nhãn đã điền | Chưa chấm độc lập, chưa đo P95 Colab/live, chưa đo hiệu quả đầu tư |

## Cách vòng bổ sung hoạt động

Sau khi Data/Research/Technical/Fundamental/Risk và các conflict rerun hoàn tất:

1. Fundamental/Risk gọi model được cấu hình để review evidence cuối.
2. Model có thể đề xuất `FETCH_FINANCIAL_HISTORY`, `VERIFY_FINANCIAL_FACT`, `FIND_RECENT_REPORT`, `CHECK_RISK_EVENT`.
3. Manager kiểm tra source allowlist, metric, kỳ không vượt as-of, ngân sách và số vòng. Nếu specialist LLM bật, dành một iteration cuối trong `max_iterations` cho nhánh này; **không cộng thêm vòng ngoài trần**.
4. Code chọn tối đa một task; Research dùng truy vấn theo đúng task/metric/kỳ, không thực thi URL/hàm do model viết.
5. Với task tìm/kiểm chứng báo cáo, mapper có thể đề xuất tối đa 4 statement facts. Chỉ T1/T2, đúng issuer, quote chính xác có label→value, đơn vị/scale/kỳ/basis/consolidation rõ ràng mới được đưa vào **bản sao runtime**. Không ghi đè provider/cache; xung đột phải review, không chọn nguồn tùy tiện.
6. Facts đã qua validator vẫn chỉ là extraction cơ học, không phải đã kiểm toán; Fundamental giảm trọng số evidence. Legacy `map_financial_facts` vẫn chỉ proposal, không trực tiếp đổi score.
7. Khi input thật thay đổi, tính lại Fundamental/Risk; Data refresh còn tính lại forecast/Technical. Review trước refresh không được dùng lại cho Manager/UI. Không có evidence mới hoặc lỗi thì giữ snapshot/kết quả trước đó.

Ngân sách vẫn có hiệu lực: demo analysis mặc định 4 LLM calls; fact mapping/review lại có thể hết quota nên trả thiếu review, không giấu bằng kết quả model cũ. Muốn demo sâu có thể đặt `STOCK_AGENT_LLM_ANALYSIS_MAX_CALLS=6` trong `.env` (trần 6); không bắt buộc và không tự thay `.env` chứa khóa. Direct graph API cũng có request scope, mặc định dùng `AgentState.request_budget`.

## Chạy và kiểm tra

Khởi động lại app, không cần cài thêm:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Ví dụ: `Phân tích FPT`; tiếp theo `Vẽ biểu đồ doanh thu theo quý trong 8 quý`; `Vẽ doanh thu và dòng tiền FPT theo năm trong 5 năm`; `Vẽ RSI và MACD FPT`.

Nếu dữ liệu không có đủ kỳ hoặc cơ sở báo cáo, app sẽ nêu lý do giữ lại phép so sánh thay vì vẽ chuỗi tưởng tượng. Coverage của Risk là mức đầy đủ checklist, không phải xác suất cổ phiếu an toàn.

### Review trên dữ liệu thật

Sau khi chat phân tích và app đã lưu snapshot:

```powershell
.\.venv\Scripts\python.exe scripts\review_real_analysis.py --limit 20 --output .runtime\evaluations\real-analysis-review-01.json
```

Trong mỗi case, giữ nguyên `frozen` và `frozen_sha256`, điền `reviewer`, `reviewed_at`, các mục `judgments` bằng `true/false`, `notes`, rồi đặt `review_status` thành `REVIEWED`. Chỉ đánh dấu supported sau khi mở nguồn và kiểm tra mã, kỳ, đơn vị và nhận định.

```powershell
.\.venv\Scripts\python.exe scripts\review_real_analysis.py --summarize .runtime\evaluations\real-analysis-review-01.json
```

Chưa có snapshot thật thì có 0 case và `PENDING_HUMAN_REVIEW`; không tự tạo case giả để lấp đủ. Công cụ bỏ fixture và request_parameters của provider; gói có thể chứa nội dung nghiên cứu của bạn nên giữ trong `.runtime` bị gitignore, không tự public. Tool không chứng thực người review độc lập; tỷ lệ rubric không phải tỷ lệ dự đoán đúng hay lợi nhuận.

## Kết quả kiểm tra đợt khắc phục

- Toàn bộ **525 tests passed** (27,64 giây), tăng 57 test so với mốc 468 trước đợt này. Gồm orchestration đầy đủ LangGraph với proposal, atomic failure/no-change, source/period/scale guards, financial normalization, Risk/KG và review pack. Dữ liệu test là fixture, không phải chất lượng nhận định tài chính đã đo.
- Benchmark synthetic `evaluate_upgrade.py --strict` qua các gate hồi quy; Recall@5 0,9625 và MRR 0,9375. Không phải independent real-world holdout.
- Streamlit AppTest: hai kết quả phân tích, 4 chart và 12 bảng; **0 exceptions**. Tắt external LLM/LSTM/network cho smoke test.
- `compileall`, `git diff --check` sạch; `.env` và review packs dưới `.runtime` bị gitignore. Không ghi snapshot fixture vào research database mặc định.
- Chưa chạy đánh giá chất lượng live trên Colab/Neo4j thật; chưa đo P95 hay rubric từ người review độc lập. Không cài thêm gói, không thay `.env`, không train/thay LSTM V2, không commit/push trong lượt này.
