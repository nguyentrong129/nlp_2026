# LAB 10 — Extractive Question Answering

LAB này dùng SQuAD 1.1, ánh xạ answer từ character span sang token span, fine-tune Transformer cho start/end prediction và đánh giá bằng Exact Match (EM) và token F1.

## Chạy trên Google Colab

1. Mở `qa_finetuning.ipynb` trong thư mục này. Bật GPU trong Colab nếu có.
2. Chạy các cell từ đầu đến cuối. Notebook tự cài thư viện, tải SQuAD, huấn luyện, đánh giá và xuất file trong `lab10/`.
3. Mở `tokenizer_alignment.ipynb` để quan sát offset và tự kiểm tra các trường hợp span đặc biệt.
4. Điền `calculations.md` và `prediction.md` **trước** khi chạy thí nghiệm. Hai file đó được để lại cho sinh viên tạo.
5. Sau khi chạy, tự xem ví dụ lỗi và hoàn thiện diễn giải trong `error_analysis.md` và `reflection.md`.

Chạy từ thư mục `lab10/` để đường dẫn đầu ra đúng. Trên Colab, nên clone repository rồi `cd` vào `nlp_2026/lab10` trước khi mở notebook, hoặc chép toàn bộ `lab10/` vào cùng thư mục làm việc. `qa_model.py` phải ở cạnh notebook.

## File

| File | Nội dung |
|---|---|
| `qa_model.py` | Alignment, hậu xử lý start/end, baseline và hàm đánh giá. |
| `qa_finetuning.ipynb` | Fine-tuning, ba độ dài context, EM/F1, answer length, lexical overlap và xuất kết quả. |
| `tokenizer_alignment.ipynb` | Quan sát token, `sequence_ids`, offsets, overflow và các tình huống alignment. |
| `document_qa.py` | Trả lời từ một context/tài liệu được cung cấp, không truy xuất tài liệu. |
| `results.csv`, `learning_curves.png`, `em_f1_analysis.png`, `answer_length_analysis.png` | Được notebook tạo từ lần chạy thực tế. |
| `error_analysis.md` | Bảng lỗi sinh từ dự đoán thật; sinh viên bổ sung giải thích. |
| `reflection.md` | Khung cho phần tự phản ánh 300–500 từ và learning check. |

Mặc định notebook dùng `distilbert-base-uncased` với 600/120/200 mẫu train/validation/test trên GPU, hoặc BERT nhỏ pretrained của Google với 120/40/100 mẫu trên CPU để chạy được trong môi trường hạn chế. Tăng `TRAIN_LIMIT`, `VALID_LIMIT`, `TEST_LIMIT`, `EPOCHS` để làm thí nghiệm lớn hơn. Tất cả độ dài 128/256/384 dùng cùng tập dữ liệu, model khởi tạo và các siêu tham số khác. SQuAD 1.1 không có public test labels, vì vậy notebook chia phần validation gốc thành validation và test cố định; test được giữ riêng khỏi lựa chọn checkpoint.

`results.csv` và ba biểu đồ trong thư mục đã được tạo từ lần chạy thật của `qa_finetuning.ipynb`; chạy lại notebook sẽ cập nhật chúng. Không có số liệu mẫu hoặc số liệu giả. Các ghi chú phân tích do sinh viên tự hoàn thiện sau khi xem kết quả.

Lần chạy kiểm tra trong repo này dùng CPU, `google/bert_uncased_L-2_H-128_A-2`, 120/40/100 mẫu, seed 42, batch size 8, learning rate 2e-5, stride 64 và 2 epoch. Kết quả model tại `max_length=384` chỉ đạt EM 0% và F1 khoảng 4.9%, thấp hơn baseline. Đây là kết quả thực của cấu hình nhỏ, chưa phải kết quả khi huấn luyện đủ dữ liệu trên GPU. Notebook in cấu hình tái lập ở cell cuối.

## Document QA

Chương trình tự tải một model QA pretrained khi chạy. Có thể truyền `--model` để dùng checkpoint riêng:

```bash
python document_qa.py --context-file article.txt --question "What is the main topic?"
```

Context phải có câu trả lời trong văn bản; LAB 10 không làm retrieval hoặc xử lý no-answer của SQuAD 2.0.

## AI assistance

- Tool: OpenAI Codex.
- Purpose: Hỗ trợ dựng cấu trúc file, boilerplate huấn luyện, code tokenizer và trực quan hóa.
- Generated: `qa_model.py`, `qa_finetuning.ipynb`, `tokenizer_alignment.ipynb`, `document_qa.py`, khung báo cáo.
- Modified: Sinh viên điền sau khi kiểm tra và chỉnh sửa.
- Verified: Sinh viên ghi lại cấu hình, lần chạy thực tế, các ví dụ tính tay và kiểm chứng kết quả.
