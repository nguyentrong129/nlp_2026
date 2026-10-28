# LAB 06 — Attention Mechanism

## Mục tiêu
- Hiểu và cài đặt cơ chế Attention bằng NumPy.
- Thực hiện Scaled Dot-Product Attention: tính scores, softmax, attention weights và context vector.
- Tìm hiểu Self-Attention, ảnh hưởng của scaling/temperature và độ dài chuỗi.
- Trực quan hóa attention weights bằng biểu đồ/heatmap.
- Thử nghiệm mô hình phân loại cảm xúc sử dụng Self-Attention và xem xét các dự đoán sai.

## Nội dung chính
1. Cài đặt softmax ổn định về số.
2. Cài đặt Scaled Dot-Product Attention và Self-Attention.
3. Kiểm tra kết quả tính toán và tính ổn định số.
4. Thí nghiệm phân phối attention, scaling và temperature.
5. Vẽ ma trận Self-Attention.
6. So sánh chi phí tính toán khi độ dài chuỗi tăng.
7. Thử nghiệm phân loại cảm xúc trên tập dữ liệu IMDB.
8. Đánh giá kết quả và ghi lại các trường hợp dự đoán sai.

## Cấu trúc
- `attention_numpy.py`: các hàm softmax, Scaled Dot-Product Attention và Self-Attention bằng NumPy.
- `self_attention.ipynb`: notebook chạy các thí nghiệm, trực quan hóa và mô hình phân loại cảm xúc.
- `attention_heatmap.png`: heatmap được tạo khi chạy notebook (nếu cell tương ứng chạy thành công).
- `confusion_matrix.png`: ma trận nhầm lẫn được tạo sau khi đánh giá mô hình (nếu cell tương ứng chạy thành công).
- `results.csv`: kết quả đánh giá mô hình sau khi notebook chạy đến bước xuất kết quả.
- `outputs/`: các biểu đồ và bảng kết quả phụ được notebook tạo ra.

## Cách chạy
1. Mở thư mục `lab06/`.
2. Mở `self_attention.ipynb` bằng Jupyter Notebook hoặc Google Colab.
3. Cài các thư viện còn thiếu nếu cần:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn torch datasets
   ```
4. Chạy các cell theo thứ tự từ trên xuống.
5. Phần tải tập IMDB cần kết nối Internet và thư viện `datasets`.
6. Sau khi chạy xong, kiểm tra các bảng, biểu đồ và kết quả được tạo trong `outputs/` và thư mục `lab06/`.

## Ghi chú
- Các kết quả trong `results.csv` và hình ảnh chỉ xuất hiện sau khi chạy thành công những cell liên quan; không xem các tệp kết quả chưa được tạo là kết quả thực nghiệm.
- Khi báo cáo, mô tả dựa trên kết quả thực tế của lần chạy.
- Nếu có yêu cầu về tính toán tay, dự đoán trước thí nghiệm, diễn giải heatmap hoặc error analysis cá nhân, hãy tự hoàn thiện theo hướng dẫn của đề bài.
