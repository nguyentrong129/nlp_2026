Prediction 1: Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ lớn hơn vì Softmax ưu tiên score cao.

Prediction 2: Nếu tất cả scores bằng nhau, attention weights bằng nhau. Với s=[1,1,1,1]: alpha=[0.25,0.25,0.25,0.25]

Prediction 3: Với s=[1,1,10], attention sẽ tập trung chủ yếu vào key có score 10.

Prediction 4: Khi sequence tăng từ 10 lên 1000 tokens, chi phí tính Self-Attention tăng khoảng 100^2/10^2=10000 lần nếu xét phần tính toán theo T^2.

Prediction 5: Attention không phải lúc nào cũng chọn đúng token quan trọng nhất; nó phụ thuộc vào dữ liệu, cách huấn luyện và mục tiêu của mô hình.