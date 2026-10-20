# LAB 04 — Text Classification (IMDb Sentiment)

- Sinh viên: Nguyễn Quý Trọng — MSSV 22001290
- Môn: Xử lý ngôn ngữ tự nhiên và ứng dụng, HK I - 2026
- Dataset: IMDb (`stanfordnlp/imdb`), bài toán sentiment nhị phân (0 = neg, 1 = pos)

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `classification.ipynb` | Notebook chạy toàn bộ thí nghiệm |
| `preprocessing.py` | Hàm load/split, kiểm tra dữ liệu, TF-IDF, Average Word2Vec |
| `results.csv` | Bảng so sánh 4 mô hình (trên validation) |
| `confusion_matrix.png` | Confusion matrix của mô hình tốt nhất (TF-IDF + SVM, validation) |
| `prediction.md` | Dự đoán trước experiment (sinh viên tự điền) |
| `error_analysis.md` | Các ví dụ đúng/sai (sinh viên tự phân tích) |
| `reflection.md` | Reflection (sinh viên tự viết) |

## Dữ liệu và split

- Train gốc 25,000 review chia stratified (seed=42): **Train 20,000 / Validation 5,000**; Test 25,000 giữ nguyên.
- Số lớp: 2 (`neg`, `pos`).
- Phân phối lớp: Train 10,000 neg / 10,000 pos; Validation 2,500 / 2,500 (cân bằng).
- Missing text: 0; missing label: 0 (kiểm tra trên train).
- Duplicate trong train: 56 (20,000 → 19,944 văn bản duy nhất), chưa loại bỏ.
- Độ dài (số từ, train): min 10, max 2,470, trung bình 234.11.

## Pipeline (không leakage)

`split → fit TF-IDF / Word2Vec chỉ trên TRAIN → transform Validation/Test → train → evaluate`

## Kết quả (validation)

| Model | Accuracy | Precision | Recall | F1 | Train time (s) | Feature size |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.8780 | 0.8694 | 0.8896 | 0.8794 | 0.81 | 10,000 |
| TF-IDF bigram + LR | 0.8852 | 0.8818 | 0.8896 | 0.8857 | 0.84 | 10,000 |
| TF-IDF + Linear SVM | 0.8860 | 0.8851 | 0.8872 | 0.8861 | 1.36 | 10,000 |
| Average Word2Vec + LR | 0.7954 | 0.7918 | 0.8016 | 0.7967 | 1.61 | 100 |

Cấu hình: `TfidfVectorizer(max_features=10000)` (bigram: `ngram_range=(1,2)`, cũng giới hạn 10,000 nên vocabulary size bằng nhau); `LogisticRegression(max_iter=1000)`; `LinearSVC()` mặc định; Word2Vec `vector_size=100, window=5, min_count=2, seed=42`, huấn luyện trên train.

### Confusion matrix (validation)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| TF-IDF + LR | 2166 | 334 | 276 | 2224 |
| TF-IDF + SVM (tốt nhất theo F1) | 2212 | 288 | 282 | 2218 |

Số review dự đoán đúng / sai của TF-IDF + LR: 4,390 / 610.

### Đánh giá trên test (TF-IDF + LR, chạy một lần sau khi chọn mô hình)

Accuracy 0.8816 · Precision 0.8811 · Recall 0.8824 · F1 0.8817

## Top features (TF-IDF + LR)

- Positive: great (6.79), excellent (5.31), wonderful (4.66), best (4.59), perfect (4.44), amazing (3.66), well (3.48), loved (3.44), enjoyed (3.43), favorite (3.33), brilliant (3.23), superb (3.21), love (3.11), today (3.08), highly (3.08)
- Negative: worst (-8.23), bad (-7.28), waste (-5.97), awful (-5.81), boring (-5.68), poor (-4.64), terrible (-4.33), nothing (-4.29), dull (-4.12), no (-4.04), worse (-3.99), poorly (-3.86), horrible (-3.83), stupid (-3.78), unfortunately (-3.69)

## AI assistance statement

- Tool: Claude (Anthropic)
- Purpose: trích xuất kết quả từ notebook và tổng hợp các file deliverable (README, preprocessing.py, results.csv, confusion_matrix.png, khung cho các file markdown)
- Generated: README.md, preprocessing.py (tổ chức lại code từ notebook), khung `prediction.md` / `error_analysis.md` / `reflection.md`
- Modified: ______ (sinh viên điền)
- Verified: ______ (sinh viên điền sau khi tự kiểm tra)

Lưu ý theo AI policy của lab: phần dự đoán, diễn giải metrics, chọn nguyên nhân lỗi và reflection do sinh viên tự viết; AI chỉ dựng khung và điền số liệu.
