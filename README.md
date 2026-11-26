# Dự báo điện năng tiêu thụ của hộ gia đình dựa trên dữ liệu đo theo thời gian

Dự án học phần Học máy, Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU).
Nhóm: Trần Trọng Hoàn (nhóm trưởng), Vũ Ngọc Quý. Giảng viên hướng dẫn: TS. Nguyễn Tuấn Anh.

Bài toán: hồi quy có giám sát, dự báo điện năng (kWh) của giờ $t+h$ với $h \in \{1, 6, 12, 24\}$
từ dữ liệu đo theo phút của một hộ gia đình (UCI *Individual Household Electric Power Consumption*,
DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54), giấy phép CC BY 4.0).

## Cấu trúc

```
cao_phim/
├── notebooks/
│   ├── 01_eda.ipynb                  # Tuần 4: khảo sát, chất lượng dữ liệu, tạo bảng điện năng theo giờ
│   ├── 02_chuan_bi_du_lieu.ipynb     # Tuần 4: bảng có giám sát, chia theo thời gian, pipeline tiền xử lý
│   ├── 03_huan_luyen_danh_gia.ipynb  # Tuần 6: mô hình cơ sở, Linear Regression, Random Forest, XGBoost, kiểm định chéo
│   └── 04_toi_uu_cai_tien.ipynb      # Tuần 8: đường cong học tập/kiểm định, đặc trưng mới, tối ưu siêu tham số, ablation, tập kiểm tra 2010
├── src/                              # mã dùng chung: đặc trưng, mô hình, đánh giá
├── ket_qua/                          # bảng kết quả do notebook 03, 04 ghi ra (CSV, Parquet, JSON)
├── bao_cao/                          # báo cáo LaTeX
└── data/                             # dữ liệu (không đưa lên Git; notebook 01 tự tải)
```

## Cài đặt

Python 3.11. Trong thư mục `cao_phim`:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Chạy lại

Chạy lần lượt bốn notebook trong thư mục `notebooks/` (kernel dùng môi trường `.venv` ở trên):

1. `01_eda.ipynb`: lần đầu sẽ tự tải tệp nén (khoảng 20 MB) từ UCI vào `data/raw/`, rồi tạo
   `data/processed/minute.parquet` và `data/processed/hourly.parquet`.
2. `02_chuan_bi_du_lieu.ipynb`.
3. `03_huan_luyen_danh_gia.ipynb` (khoảng 2 phút trên máy 16 luồng): ghi kết quả vào `ket_qua/`.
4. `04_toi_uu_cai_tien.ipynb` (khoảng 40 phút trên máy 16 luồng, phần lớn là tìm kiếm siêu tham số; giảm `N_ITER_RF`, `N_ITER_XGB` ở ô tham số nếu muốn chạy nhanh): ghi kết quả vào `ket_qua/`.

Có thể chạy không cần giao diện:

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_chuan_bi_du_lieu.ipynb 03_huan_luyen_danh_gia.ipynb 04_toi_uu_cai_tien.ipynb
```

Mọi mô hình dùng `random_state = 42`. Tập kiểm tra năm 2010 chỉ được dùng ở mục 9 của notebook 04, sau khi
cấu hình mô hình cuối đã được chọn bằng kiểm định chéo và ghi vào `ket_qua/cau_hinh_cuoi.json`.

## Báo cáo

Báo cáo nằm trong `bao_cao/`, biên dịch bằng XeLaTeX và Biber:

```bash
cd bao_cao
xelatex main && biber main && xelatex main && xelatex main
```
