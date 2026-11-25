# LAB 10 — Reflection

Sinh viên tự viết 300–500 từ sau khi hoàn tất thí nghiệm. Gợi ý: so sánh LAB 08 classification, LAB 09 token classification và LAB 10 span prediction; nêu vai trò của cùng một pretrained backbone, prediction head và target khác nhau.

```text
                     Pretrained Transformer
                       /        |        \
                  [CLS]      Token reps  Token reps
                    |            |          |
             Classification      NER     Start / End
                  LAB 08        LAB 09      LAB 10
```

## Individual learning check

1. Extractive QA khác classification như thế nào?
2. Tại sao QA cần start và end position?
3. Character offset khác token index như thế nào?
4. Tại sao cần offset mapping?
5. Stride dùng để làm gì?
6. Vì sao context dài cần chia window?
7. Exact Match đo điều gì?
8. Token F1 của QA khác accuracy như thế nào?
9. Vì sao F1 có thể lớn hơn 0 nhưng EM bằng 0?
10. Wrong start khác wrong end như thế nào?
11. Extractive QA khác RAG ở đâu?
12. Khi nào extractive QA không thể trả lời?
