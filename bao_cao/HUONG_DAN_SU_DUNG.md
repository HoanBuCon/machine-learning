# Hướng dẫn sử dụng template khóa luận tốt nghiệp

## 1. Chuẩn bị

Cài đặt các phần mềm sau:

- TeXstudio
- MiKTeX
- Biber

Sau khi cài MiKTeX, mở MiKTeX Console và cập nhật gói nếu được yêu cầu.

## 2. Mở template

1. Giải nén file template.
2. Mở TeXstudio.
3. Chọn `File > Open`.
4. Mở file `main.tex`.

## 3. Cấu trúc thư mục

- `main.tex`: file chính để biên dịch.
- `bia/`: chứa bìa ngoài và bìa trong.
- `chuong/`: chứa lời cảm ơn, lời cam đoan, mở đầu, các chương, kết luận và nhận xét giáo viên hướng dẫn.
- `tai_nguyen/`: chứa logo và các ảnh dùng trong báo cáo.
- `tham_khao/`: chứa file tài liệu tham khảo `tai_lieu.bib`.

## 4. Chỉnh thông tin cá nhân và đề tài

Sinh viên cần chỉnh các file sau:

- `bia/bia_ngoai.tex`
- `bia/bia_trong.tex`
- `chuong/loi_cam_on.tex`
- `chuong/loi_cam_doan.tex`
- `chuong/mo_dau.tex`
- `chuong/chuong_1.tex`
- `chuong/chuong_2.tex`
- `chuong/chuong_3.tex`
- `chuong/chuong_4.tex`
- `chuong/ket_luan.tex`

Các vị trí để trống như `Tên đề tài`, `Họ và tên sinh viên`, `ngày ... tháng ... năm ...` cần được thay bằng thông tin thực tế.

## 5. Chèn hình ảnh

Đưa ảnh vào thư mục `tai_nguyen/`, sau đó chèn vào file `.tex` bằng cú pháp. Trong template đã có ví dụ chèn ảnh thật tại `chuong/chuong_3.tex`.

```tex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{ten_anh.png}
\caption{Tên hình}
\label{fig:ten-hinh}
\end{figure}
```

Tham chiếu đến hình trong nội dung:

```tex
Như minh họa trong Hình~\ref{fig:ten-hinh}.
```

## 6. Chèn bảng

Ví dụ bảng:

```tex
\begin{table}[H]
\centering
\caption{Tên bảng}
\label{tab:ten-bang}
\begin{tabular}{|c|p{5cm}|p{5cm}|}
\hline
\textbf{STT} & \textbf{Nội dung} & \textbf{Ghi chú} \\
\hline
1 & Nội dung mẫu & Ghi chú mẫu \\
\hline
\end{tabular}
\end{table}
```

Tham chiếu đến bảng trong nội dung:

```tex
Kết quả được trình bày trong Bảng~\ref{tab:ten-bang}.
```

## 7. Quản lý tài liệu tham khảo

Thêm tài liệu tham khảo vào file:

```text
tham_khao/tai_lieu.bib
```

Trích dẫn trong nội dung:

```tex
Nội dung cần trích dẫn~\cite{sommerville2016software}.
```

Danh mục tài liệu tham khảo sẽ tự sinh ở cuối báo cáo.

## 8. Tạo danh sách gạch đầu dòng

Ví dụ danh sách bullet:

```tex
\begin{itemize}
    \item Nội dung thứ nhất.
    \item Nội dung thứ hai.
    \item Nội dung thứ ba.
\end{itemize}
```

## 9. Biên dịch trong TeXstudio

Trong TeXstudio, mở `main.tex`, sau đó bấm nút biên dịch màu xanh hoặc phím tắt `F5`.

Khi mới thêm mục lục, hình, bảng hoặc tài liệu tham khảo, cần bấm biên dịch vài lần để số trang, số hình, số bảng và trích dẫn được cập nhật đầy đủ.

Nếu tài liệu tham khảo chưa hiện đúng, chọn trong TeXstudio:

- `Tools > Bibliography`
- Sau đó bấm biên dịch lại bằng nút màu xanh hoặc `F5`

## 10. Dọn file biên dịch

Sau khi biên dịch, TeXstudio và MiKTeX có thể sinh ra nhiều file phụ như `.aux`, `.log`, `.toc`, `.bbl`. Có thể giữ nguyên các file này, hoặc xóa các file phụ nếu muốn thư mục gọn hơn. Không xóa các file `.tex`, `.bib`, ảnh trong `tai_nguyen/` và file `main.pdf`.
