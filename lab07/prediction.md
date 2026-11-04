Prediction 1

Prediction: Có, Transformer có thể xử lý các token song song.

Reason: Transformer xử lý nhiều token cùng lúc, còn RNN xử lý từng token một.

Prediction 2

Prediction: Không, Transformer sẽ khó nhận biết thứ tự token.

Reason: Positional Encoding giúp mô hình biết vị trí của từng token.

Prediction 3

Prediction: Không thể phân biệt rõ hai câu.

Reason: Nếu không có thông tin vị trí, mô hình chỉ nhận biết các token mà không biết thứ tự của chúng.

Prediction 4

Prediction: Không nhất thiết giống nhau.

Reason: Mỗi attention head có thể chú ý đến những token khác nhau.

Prediction 5

Prediction: Không chắc chắn accuracy sẽ tăng.

Reason: Nhiều heads hơn chưa chắc giúp mô hình dự đoán tốt hơn.

Prediction 6

Prediction: Không phải lúc nào Transformer cũng nhanh hơn RNN.

Reason: Với sequence length dài, Transformer cần tính toán attention nhiều hơn nên có thể tốn thời gian.