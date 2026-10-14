# Error analysis

Mô hình: TF-IDF + Linear SVM (F1 cao nhất trên validation). Nhãn: 0 = negative, 1 = positive.
Văn bản bên dưới là phần đầu của review (bị cắt trong output notebook). Cột Analysis để sinh viên tự điền
(gợi ý category từ đề: lexical ambiguity, negation, sarcasm, short text, rare word, domain-specific term, data noise, class imbalance).

## Confusion matrix (validation)

TN = 2212, FP = 288, FN = 282, TP = 2218 (xem `confusion_matrix.png`).

## 3 ví dụ dự đoán đúng

| Index | Text (đầu review) | True | Predicted | Analysis |
|---|---|---|---|---|
| 0 | First of all, the actor they have to play Jesu... | 0 | 0 | |
| 1 | I must have seen this a dozen times over the y... | 1 | 1 | |
| 2 | Eric Stoltz delivers an extraordinary performa... | 1 | 1 | |

## 3 ví dụ dự đoán sai

| Index | Text (đầu review) | True | Predicted | Analysis |
|---|---|---|---|---|
| 10 | It must be the most corniest TV show on the ai... | 0 | 1 | |
| 12 | I am obsessed! The story is amazing and the sh... | 1 | 0 | |
| 48 | The Angry Red Planet (Quickie Review) <br /><b... | 0 | 1 | |

## Nhận xét chung (sinh viên tự viết)

- Class nào bị nhầm nhiều nhất:
- Loại error chiếm nhiều:
- Metric phản ánh vấn đề đó:
- Representation ảnh hưởng thế nào:
