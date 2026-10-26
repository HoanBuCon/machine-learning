# BÁO CÁO BÀI KIỂM TRA 1 (BKT1 – TUẦN 4)
## HỌC PHẦN: HỌC MÁY (MACHINE LEARNING)

**Đề tài:** Dự báo điện năng tiêu thụ theo giờ của hộ gia đình (*Individual Household Electric Power Consumption*)  
**Thư mục thực hành:** `cao_phim/`  
**Notebook minh chứng:** `cao_phim/notebooks/01_eda.ipynb` và `cao_phim/notebooks/02_chuan_bi_du_lieu.ipynb`

---

## BẢNG ĐỐI CHIẾU TIÊU CHÍ ĐÁNH GIÁ (RUBRIC MAPPING)

| STT | Tiêu chí đánh giá | Trọng số | Vị trí minh chứng trong Báo cáo | Vị trí minh chứng trong Notebook / Mã nguồn |
|:---:|---|:---:|---|---|
| **TC1** | Phát biểu bài toán học máy | 1,0 | Chương 1 (Bảng 1.1) | `01_eda.ipynb` (Ô 0), `02_chuan_bi_du_lieu.ipynb` (Ô 0) |
| **TC2** | Mục tiêu và tiêu chí thành công | 1,0 | Chương 2 (Bảng 2.1) | `01_eda.ipynb` (Ô 0), Báo cáo Chương 2 |
| **TC3** | Nguồn và từ điển dữ liệu | 1,0 | Chương 3 (Bảng 3.1) | `01_eda.ipynb` (Ô 0, 4) |
| **TC4** | Khảo sát thống kê và phân bố | 1,0 | Chương 4 (Bảng 4.1, 4.2) | `01_eda.ipynb` (Ô 5–17) |
| **TC5** | Quan hệ đặc trưng – nhãn | 1,0 | Chương 5 (Bảng 5.1, 5.2) | `01_eda.ipynb` (Ô 33–39) |
| **TC6** | Phát hiện vấn đề chất lượng | 1,0 | Chương 6 (Bảng 6.1) | `01_eda.ipynb` (Ô 18–23) |
| **TC7** | Xử lý vấn đề chất lượng (Trước – Sau) | 1,0 | Chương 7 (Bảng 7.1, 7.2, 7.3) | `01_eda.ipynb` (Ô 24–32) |
| **TC8** | Chuẩn bị dữ liệu (Pipeline, Không rò rỉ) | 1,0 | Chương 8 (Bảng 8.1, 8.2) | `02_chuan_bi_du_lieu.ipynb` (Ô 4–17) |
| **TC9** | Lý giải chiến lược đánh giá và chỉ số | 1,0 [Cá nhân] | Chương 9 | Chuẩn bị vấn đáp (Phần E1 Hướng dẫn) |
| **TC10**| Thao tác thực hành tại chỗ | 1,0 [Cá nhân] | Chương 10 | Ô tham số đầu `02_chuan_bi_du_lieu.ipynb` (Ô 4, 19) |

---

## CHƯƠNG 1: PHÁT BIỂU BÀI TOÁN HỌC MÁY (TIÊU CHÍ 1)

### 1.1 Bối cảnh thực tế và Đặt vấn đề
Trong bối cảnh chuyển dịch năng lượng và triển khai lưới điện thông minh (*Smart Grid*), việc nắm bắt và dự báo sớm nhu cầu tiêu thụ điện năng tại cấp độ hộ gia đình đóng vai trò then chốt nhằm tối ưu hóa chi phí vận hành, cân bằng phụ tải đỉnh và quản lý hiệu quả các nguồn năng lượng tái tạo phân tán. Bộ dữ liệu đo đạc tại hộ gia đình ở Sceaux (Pháp) phản ánh hành vi phụ tải thực tế với chu kỳ sống động, đan xen giữa mức tiêu thụ nền ổn định và các đỉnh phụ tải ngắn hạn đột biến.

### 1.2 Bảng đặc tả bài toán học máy cụ thể

**Bảng 1.1: Phát biểu chi tiết bài toán học máy trên tập dữ liệu của nhóm**

| Thành phần | Đặc tả chi tiết theo dữ liệu của nhóm |
|---|---|
| **Loại bài toán** | **Hồi quy chuỗi thời gian (Time Series Regression)**. Đầu ra mục tiêu là đại lượng thực liên tục (điện năng tiêu thụ theo giờ, đơn vị $\text{kWh}$). Áp dụng chiến lược dự báo trực tiếp (*Direct Forecasting*) cho từng tầm dự báo $h$. |
| **Một mẫu dữ liệu** | Là một cặp gồm thời điểm gốc $t$ và tầm dự báo $h \in \{1, 6, 12, 24\}$ giờ.<br>*Ví dụ cụ thể có ngày giờ:* Tại thời điểm gốc $t = \text{20:00 ngày 03/02/2009}$, mô hình tiếp nhận các thông tin đo lường quá khứ $\le t$ để dự báo điện năng tiêu thụ trong giờ mục tiêu $t+h = \text{21:00 ngày 03/02/2009}$ ($h=1$), hoặc $t+h = \text{02:00 ngày 04/02/2009}$ ($h=6$). |
| **Đầu vào (Features)** | Chỉ sử dụng các thông tin **đã biết tại hoặc trước thời điểm $t$**:<br>1. *Đặc trưng độ trễ:* Các giá trị điện năng giờ quá khứ ($y_t, y_{t-1}, y_{t-5}, y_{t-23}, y_{t-167}$).<br>2. *Đặc trưng chu kỳ:* Điện năng cùng giờ hôm trước ($y_{t+h-24}$), cùng giờ tuần trước ($y_{t+h-168}$).<br>3. *Đặc trưng lịch thời điểm mục tiêu:* Giờ trong ngày (`hour` $\in [0, 23]$), thứ trong tuần, tháng. |
| **Đầu ra mục tiêu (Target)** | $y_{t+h} = \text{energy\_kwh}$: Điện năng tác dụng tiêu thụ trong trọn vẹn giờ $t+h$.<br>- *Đơn vị:* Kilowatt-giờ ($\text{kWh}$).<br>- *Miền giá trị thực tế:* $[0,124; 6,561]\text{ kWh}$.<br>- *Công thức tính:* $E_{\text{giờ}} = \overline{P}_{\text{giờ}} \times 1\text{ h}$, trong đó $\overline{P}_{\text{giờ}}$ là công suất tác dụng trung bình các phút có đo trong giờ (yêu cầu $\ge 54$ phút hợp lệ). |
| **Số lượng mẫu** | Với $h = 1$: Tập Train có **26.523** mẫu có nhãn (mục tiêu trước ngày 01/01/2010); Tập Test có **7.620** mẫu có nhãn (năm 2010). |
| **Biến cấm sử dụng làm đầu vào** | `Global_intensity` (tương quan 0,999 với công suất) và `Voltage` đo đồng thời tại $t+h$. Tại thời điểm $t$, các giá trị này ở tương lai chưa tồn tại; việc đưa vào sẽ gây hiện tượng **rò rỉ dữ liệu (Data Leakage)** nghiêm trọng. |

---

## CHƯƠNG 2: MỤC TIÊU VÀ TIÊU CHÍ THÀNH CÔNG (TIÊU CHÍ 2)

### 2.1 Bảng ánh xạ Mục tiêu thực tế $\to$ Mục tiêu học máy $\to$ Chỉ số đánh giá

**Bảng 2.1: Mục tiêu và tiêu chí thành công định lượng**

| Mục tiêu thực tế | Mục tiêu học máy | Chỉ số và Mức chuẩn đối sánh (Baseline) | Mức mong muốn đạt được |
|---|---|---|---|
| Biết trước lượng điện sắp dùng trong giờ tới để tối ưu điều khiển phụ tải (máy sưởi, bình nước nóng). | Hồi quy $y_{t+1}$ từ chuỗi quan sát $\le t$ ($h=1$). | **Chỉ số chính: MAE (kWh)**<br>- *Naive Baseline* ($\hat{y}_{t+1} = y_t$): $\text{MAE} \approx 0,449\text{ kWh}$.<br>- *Dự báo trung bình Train*: $\text{MAE} \approx 0,750\text{ kWh}$. | $\text{MAE} \le 0,404\text{ kWh}$ (**giảm ít nhất 10%** so với Naive Baseline tốt nhất). |
| Lập kế hoạch vận hành năng lượng trong ngày ($h \in \{6, 12, 24\}$). | Hồi quy $y_{t+h}$, mỗi tầm $h$ một mô hình riêng. | **Chỉ số chính: MAE (kWh)**<br>- *Seasonal Naive 24* ($h=24$): $\text{MAE} \approx 0,664\text{ kWh}$. | $\text{MAE} \le 0,598\text{ kWh}$ (**giảm ít nhất 10%** so với Seasonal Naive ở mọi tầm $h$). |
| Tránh dự báo lệch có hệ thống giữa các mùa và khung giờ. | Đảm bảo mô hình không thiên lệch (*unbiased*). | **Chỉ số phụ: Bias trung bình**<br>$\text{Mean Error} = \frac{1}{N}\sum (\hat{y} - y)$. | $|\text{Bias}| < 0,05\text{ kWh}$. |
| Kiểm soát rủi ro tại các giờ phụ tải đỉnh đột biến. | Phạt nặng các sai số lớn ở giờ cao điểm. | **Chỉ số phụ: RMSE (kWh)** | Báo cáo song song cùng MAE; đánh giá riêng sai số khung 19–21h. |

### 2.2 Rationale lựa chọn các chỉ số đánh giá (Tiêu chí 9)
1. **MAE (Mean Absolute Error) – Chỉ số chính:** MAE đo khoảng cách sai lệch trung bình trên cùng thang đo vật lý của nhãn ($\text{kWh}$), có tính chất tuyến tính và ít bị chi phối cực đoan bởi một vài đỉnh công suất đột biến nhưng có thật.
2. **RMSE (Root Mean Squared Error) – Chỉ số phụ:** Bổ trợ cho MAE để phát hiện xem mô hình có mắc các lỗi dự báo quá lớn ở các giờ cao điểm hay không.
3. **MAPE (Mean Absolute Percentage Error) – Chỉ mang tính tham khảo:** Dù điện năng giờ tối thiểu là $0,124\text{ kWh} > 0$ nên mẫu số không bị chia cho 0, nhưng có khoảng $1\%$ số giờ có mức tiêu thụ $< 0,17\text{ kWh}$. Tại các giá trị nền thấp này, sai số tuyệt đối dù rất nhỏ (ví dụ $0,1\text{ kWh}$) cũng đẩy MAPE lên gần $100\%$, gây méo mó đánh giá tổng thể.
4. **MASE (Mean Absolute Scaled Error) – Chỉ số chuẩn hóa:** So sánh sai số của mô hình với sai số của mô hình cơ sở Seasonal Naive trên tập huấn luyện, giúp đánh giá độc lập thang đo.

---

## CHƯƠNG 3: NGUỒN VÀ TỪ ĐIỂN DỮ LIỆU (TIÊU CHÍ 3)

### 3.1 Thông tin pháp lý và Nguồn dữ liệu
* **Tên tập dữ liệu:** *Individual Household Electric Power Consumption*.
* **Cơ quan chủ quản & Lưu trữ:** UCI Machine Learning Repository.
* **Đường dẫn truy cập:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
* **Mã định danh số (DOI):** [10.24432/C58K54](https://doi.org/10.24432/C58K54).
* **Tác giả:** Georges Hébrail, Alice Bérard (Électricité de France - EDF R&D).
* **Giấy phép sử dụng:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Quy mô thu thập:** Một hộ gia đình duy nhất tại thị trấn Sceaux (nằm cách trung tâm Paris khoảng 7 km về phía nam, Pháp). Dữ liệu được đo đều đặn mỗi phút một lần từ ngày 16/12/2006 17:24 đến ngày 26/11/2010 21:02 (tổng cộng 2.075.259 phút đo liên tục, gồm 9 thuộc tính).

### 3.2 Bảng từ điển dữ liệu (Data Dictionary)

**Bảng 3.1: Từ điển thuộc tính dữ liệu thô và sau chuẩn hóa**

| Tên thuộc tính | Ý nghĩa vật lý | Đơn vị | Kiểu thô | Kiểu xử lý | Miền giá trị thực tế | Vai trò trong bài toán |
|---|---|:---:|:---:|:---:|:---:|---|
| `Date` | Ngày đo đạc (dd/mm/yyyy) | – | `object` | `datetime64` | 16/12/2006 – 26/11/2010 | Thành phần lập chỉ mục thời gian |
| `Time` | Thời gian đo đạc (hh:mm:ss) | – | `object` | `datetime64` | 00:00:00 – 23:59:00 | Thành phần lập chỉ mục thời gian |
| `Global_active_power` | Công suất tác dụng trung bình phút | kW | `object` | `float64` | 0,076 – 11,122 kW | **Nguồn nhãn:** Quy đổi sang điện năng giờ ($E = \overline{P} \times 1\text{h}$) |
| `Global_reactive_power`| Công suất phản kháng trung bình phút | kvar | `object` | `float64` | 0,000 – 1,390 kvar | Thống kê phụ (23,5% bằng 0) |
| `Voltage` | Điện áp hiệu dụng trung bình phút | V | `object` | `float64` | 223,20 – 254,15 V | Đo đồng thời $\implies$ Loại bỏ khỏi đầu vào |
| `Global_intensity` | Cường độ dòng điện trung bình phút | A | `object` | `float64` | 0,200 – 48,400 A | **CẤM DÙNG:** Tương quan 0,999 với GAP $\implies$ Rò rỉ |
| `Sub_metering_1` | Năng lượng khu vực bếp | Wh/phút | `object` | `float64` | 0,000 – 88,000 Wh/phút | Đặc trưng phụ trễ quá khứ |
| `Sub_metering_2` | Năng lượng khu giặt, sấy, tủ lạnh | Wh/phút | `object` | `float64` | 0,000 – 80,000 Wh/phút | Đặc trưng phụ trễ quá khứ |
| `Sub_metering_3` | Năng lượng bình nước nóng và điều hòa | Wh/phút | `float64`| `float64` | 0,000 – 31,000 Wh/phút | Đặc trưng phụ trễ quá khứ |

---

## CHƯƠNG 4: THỐNG KÊ MÔ TẢ VÀ PHÂN BỐ DỮ LIỆU (TIÊU CHÍ 4)

### 4.1 Bảng thống kê mô tả mở rộng (7 biến đo theo phút)

**Bảng 4.1: Thống kê mô tả 7 biến đo liên tục (trên 2.049.280 phút hợp lệ)**

| Thuộc tính | Count | Mean | Std | Min | 25% | 50% | 75% | Max | Skew | % Bằng 0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Global_active_power` (kW) | 2.049.280 | 1,092 | 1,057 | 0,076 | 0,308 | 0,602 | 1,528 | 11,122 | +1,786 | 0,00% |
| `Global_reactive_power` (kvar)| 2.049.280 | 0,124 | 0,113 | 0,000 | 0,048 | 0,100 | 0,194 | 1,390 | +1,262 | 23,53% |
| `Voltage` (V) | 2.049.280 | 240,84 | 3,24 | 223,20 | 238,99 | 241,01 | 242,89 | 254,15 | -0,327 | 0,00% |
| `Global_intensity` (A) | 2.049.280 | 4,628 | 4,444 | 0,200 | 1,400 | 2,600 | 6,400 | 48,400 | +1,849 | 0,00% |
| `Sub_metering_1` (Wh/phút) | 2.049.280 | 1,122 | 6,153 | 0,000 | 0,000 | 0,000 | 0,000 | 88,000 | +5,945 | 91,75% |
| `Sub_metering_2` (Wh/phút) | 2.049.280 | 1,299 | 5,822 | 0,000 | 0,000 | 0,000 | 1,000 | 80,000 | +7,091 | 70,11% |
| `Sub_metering_3` (Wh/phút) | 2.049.280 | 6,458 | 8,437 | 0,000 | 0,000 | 1,000 | 17,000 | 31,000 | +0,725 | 41,61% |

**Nhận xét cốt lõi:**
1. Cả 7 biến đo đều có đúng 2.049.280 giá trị có đo; sự bằng nhau tuyệt đối này gợi ý dữ liệu thiếu theo cụm đồng thời (được kiểm chứng tại Mục 6).
2. `Global_active_power` có trung bình $1,092\text{ kW}$ lớn hơn nhiều so với trung vị $0,602\text{ kW}$, hệ số lệch $\text{skew} = 1,786$. Đúng $75\%$ số phút đo dưới $1,528\text{ kW}$, đỉnh cực đại đạt $11,122\text{ kW}$.
3. Phụ tải nền tối thiểu luôn là $0,076\text{ kW}$ (không bao giờ bằng 0).
4. `Sub_metering_3` (bình nước nóng & điều hòa) có mức tiêu thụ trung bình $6,458\text{ Wh/phút}$, **gấp 2,67 lần** tổng hai nhóm còn lại gộp lại ($6,458 / (1,122 + 1,299) \approx 2,667$) và gấp 5,76 lần riêng nhóm bếp ($6,458 / 1,122 \approx 5,756$). Nhóm này chiếm $35,5\%$ tổng điện tiêu thụ, bếp chiếm $6,2\%$, giặt sấy $7,1\%$, còn lại **$51,2\%$ điện năng thuộc về các thiết bị không gắn công tơ riêng** (ví dụ có thể là hệ thống đèn, tivi, máy tính, lò sưởi).
5. Phân tích giờ bật bếp (`Sub_metering_1 > 0`, sau khi `dropna()`): Tỷ lệ hoạt động cao nhất lúc 21h (27,5%), 20h (19,3%), 8h (14,5%), 14h (14,2% – có thể là máy rửa bát sau bữa trưa); thấp nhất lúc 4–6h sáng (0,2–0,4%).

### 4.2 Thống kê và Phân bố của Nhãn mục tiêu theo giờ (`energy_kwh`)

**Bảng 4.2: Thống kê mô tả nhãn điện năng theo giờ**

| Chỉ số thống kê | Toàn bộ chuỗi theo giờ | Tập TRAIN (< 2010-01-01) |
|---|:---:|:---:|
| Tổng số mốc giờ trên lưới | 34.589 giờ | 26.903 giờ |
| Số giờ hợp lệ ($\ge 54$ phút đo) | 34.143 giờ ($98,71\%$) | 26.523 giờ ($98,59\%$) |
| Số giờ thiếu (gán `NaN`) | 446 giờ ($1,29\%$) | 380 giờ ($1,41\%$) |
| Trung bình ($\text{kWh}$) | 1,092 | 1,100 |
| Độ lệch chuẩn ($\text{kWh}$) | 0,898 | 0,904 |
| Trung vị ($\text{kWh}$) | 0,782 | 0,794 |
| Min / Max ($\text{kWh}$) | 0,124 / 6,561 | 0,124 / 6,561 |
| Hệ số lệch Skew | +1,30 | +1,28 |
| Phân vị 1% ($Q_{0,01}$) | 0,170 | 0,173 |

**Nhận xét phân bố nhãn:**
* Nhãn có dạng đa đỉnh: Đỉnh cao nhất tập trung ở vùng phụ tải nền khoảng $0,23\text{ kWh}$ (~5.350 giờ), đỉnh thứ hai xuất hiện ở vùng sinh hoạt khoảng $1,4\text{ kWh}$ (~1.300 giờ).
* Tính chu kỳ ngày đêm: Thấp nhất vào 3–4h sáng (trung vị $\approx 0,32\text{ kWh}$), đạt đỉnh vào 20h tối (trung vị $\approx 1,90\text{ kWh}$). Khung giờ 19–21h có độ lệch chuẩn lớn nhất (20h: 1,176; 19h: 1,154; 21h: 1,102), là vùng biến động khó dự báo nhất.
* Mùa vụ: Cuối tuần có trung vị tiêu thụ ($1,014$ và $0,904\text{ kWh}$) cao hơn rõ so với ngày thường ($0,62 - 0,82\text{ kWh}$). Tháng 12 tiêu thụ cao nhất (trung vị $1,398\text{ kWh}$ do thời tiết lạnh), tháng 8 thấp bất thường (trung vị $0,289\text{ kWh}$, có thể do gia đình đi nghỉ hè).
* Tính mất cân bằng: Trong bài toán hồi quy, có khoảng $2,12\%$ số giờ vượt qua ngưỡng râu trên IQR ($3,44\text{ kWh}$).

---

## CHƯƠNG 5: QUAN HỆ ĐẶC TRƯNG – NHÃN VÀ TỰ TƯƠNG QUAN (TIÊU CHÍ 5)

### 5.1 Ma trận tương quan đặc trưng gốc trên tập Train

**Bảng 5.1: Hệ số tương quan với điện năng theo giờ (`energy_kwh`) trên Train**

| Thuộc tính gốc | Tương quan Pearson ($r$) | Tương quan Spearman ($\rho$) | Nhận xét ý nghĩa mô hình |
|---|:---:|:---:|---|
| `Global_intensity` | +0,999 | +0,999 | **Rò rỉ dữ liệu tuyệt đối:** Đo đồng thời với target $\implies$ Cấm dùng. |
| `Sub_metering_3` | +0,687 | +0,724 | Tương quan thuận mạnh nhất trong các nhóm phụ tải. |
| `Sub_metering_1` | +0,499 | +0,340 | Bếp bật đẩy công suất tổng tăng vọt. |
| `Sub_metering_2` | +0,450 | +0,358 | Khu vực giặt sấy có tính gián đoạn cao. |
| `Voltage` | -0,375 | -0,381 | Sụt áp lưới khi tổng công suất tiêu thụ gia tăng. |
| `Global_reactive_power` | +0,326 | +0,268 | Tương quan thuận yếu. |

### 5.2 Tương quan đặc trưng trễ, đặc trưng lịch và Hàm tự tương quan ACF

**Bảng 5.2: Tương quan Spearman với target ($h=1$) và Hệ số tự tương quan ACF trên Train**

| Độ trễ / Đặc trưng | Tương quan Spearman với Target ($h=1$) | Hệ số tự tương quan ACF chuỗi Train | Ý nghĩa kiến trúc mô hình |
|---|:---:|:---:|---|
| `lag_0` ($t$) | +0,764 | 1,000 (Lag 0) | Đặc trưng dự báo mạnh nhất ở tầm ngắn. |
| `lag_1` ($t-1$) | +0,534 | +0,719 (Lag 1) | Quán tính tiêu thụ ngắn hạn. |
| `lag_5` ($t-5$) | +0,026 | +0,023 (Lag 6) | Tương quan suy giảm tiệm cận 0 sau 5–6 giờ. |
| `lag_23` ($t-23$) | +0,516 | +0,433 (Lag 24) | **Hồi phục chu kỳ ngày:** Cùng giờ hôm qua tương quan rất mạnh. |
| `lag_167` ($t-167$) | +0,512 | +0,465 (Lag 168) | **Hồi phục chu kỳ tuần:** Cùng giờ tuần trước tương quan rất mạnh. |
| `hour` ($t+h$) | +0,347 | – | Quan hệ phi tuyến mạnh giữa giờ trong ngày và mức tiêu thụ. |

**Bốn nhận xét định hình thiết kế mô hình máy học:**
1. **Suy giảm ngắn hạn và Phục hồi mùa vụ:** ACF giảm dốc từ 0,719 (lag 1) về 0,023 (lag 6), nhưng bật tăng mạnh lên 0,433 (lag 24) và 0,465 (lag 168). Ở tầm dự báo xa ($h \ge 6$), các lag quá khứ gần mất tác dụng, mô hình bắt buộc phải khai thác lag chu kỳ 24h và 168h.
2. **Quan hệ phi tuyến của biến lịch:** Giờ trong ngày có phân bố hai đỉnh phi tuyến. Với mô hình tuyến tính (Ridge), bắt buộc mã hóa One-Hot hoặc Fourier sin/cos; với mô hình phi tuyến (XGBoost, LightGBM), mô hình tự chia nhánh nắm bắt quan hệ này.
3. **Loại bỏ nguy cơ rò rỉ:** Loại bỏ hoàn toàn `Global_intensity` và `Voltage` đồng thời.
4. **Hiện tượng đa cộng tuyến:** Các biến trễ chu kỳ có tương quan nội bộ cao (`lag_0` và `lag_1` tương quan 0,705; `lag_23` và `lag_167` tương quan 0,477). Mô hình tuyến tính bắt buộc áp dụng điều chuẩn Ridge L2.

---

## CHƯƠNG 6: PHÁT HIỆN VÀ ĐỊNH LƯỢNG VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU (TIÊU CHÍ 6)

**Bảng 6.1: Bảng tổng hợp định lượng 9 vấn đề chất lượng dữ liệu**

| STT | Vấn đề chất lượng | Số lượng đo đạc | Tỷ lệ định lượng | Rủi ro kỹ thuật nếu không xử lý |
|:---:|---|---|---|---|
| 1 | Ký tự lạ sai định dạng | 155.874 ô ký tự `?` | 25.979 dòng $\times$ 6 cột | Ép kiểu 6 cột thành `object`, làm sập toàn bộ phép tính toán số học. |
| 2 | Bản ghi trùng lặp mốc giờ | 0 bản ghi trùng | 0,00% | Không có rủi ro trùng lặp. |
| 3 | Mốc thời gian bị gián đoạn | 0 mốc thời gian thiếu | 0,00% | Lưới thời gian liên tục 100%, bảo đảm tính đều đặn của chuỗi. |
| 4 | Dữ liệu khuyết thiếu (NaN) | 25.979 dòng | 1,252% tổng số dòng | Toàn bộ 25.979 dòng thiếu trọn vẹn cả 7 biến (thiếu một phần: 0 dòng). |
| 5 | Cấu trúc thiếu dạng khối | 71 khối thiếu; 6 khối > 1 ngày | 94,9% số phút thiếu nằm ở khối dài | Khối dài nhất 120,4 giờ (17/08/2010). Không thể nội suy vì sẽ tạo dữ liệu giả rò rỉ tương lai. |
| 6 | Ngoại lai theo 1,5·IQR | 94.907 phút (GAP); SM1: 8,25% | GAP: 4,63%; SM3: 0% | Là các đỉnh tiêu thụ có thật trong sinh hoạt, giữ lại để mô hình học dự báo đỉnh. |
| 7 | Tính hợp lý vật lý | 0 giá trị âm; 57 phút điện áp > 253V | Áp cao: 0,003% | 253V = 230V + 10% (tiêu chuẩn điện áp châu Âu NF EN 50160). Cảm biến vật lý hoạt động chính xác. |
| 8 | Nhất quán năng lượng | e_other < 0 ở 0,05% số phút | Min = -2,4 Wh/phút | Lệch xung nhịp nhỏ giữa các cảm biến thành phần, sai số vi mô không đáng kể. |
| 9 | Nguy cơ rò rỉ dữ liệu | Tương quan $r = 0,99909$ giữa GAP và $V \cdot I$ | ~ 0,9991 | Cường độ dòng điện đo cùng lúc với nhãn, nếu dùng làm đặc trưng sẽ rò rỉ đáp án. |

---

## CHƯƠNG 7: XỬ LÝ VẤN ĐỀ CHẤT LƯỢNG VÀ SO SÁNH TRƯỚC – SAU (TIÊU CHÍ 7)

### 7.1 Bảng quyết định xử lý chất lượng dữ liệu

**Bảng 7.1: Bảng quyết định xử lý và lý do kỹ thuật**

| Vấn đề | Quyết định xử lý | Lý do kỹ thuật và Phương án loại trừ |
|---|---|---|
| Ký tự lạ `?` | `read_csv(na_values=["?", ""])` + ép `float64` | Xử lý đúng quy ước UCI. *Loại trừ:* Không thay bằng 0 vì hộ luôn có tải nền tối thiểu $0,076\text{ kW}$. |
| Khuyết thiếu (Target) | Gộp theo giờ với ngưỡng $\ge 54$ phút; thiếu gán `NaN` | *Loại trừ nội suy:* Các khối thiếu dài 1–5 ngày, nội suy sẽ lấy dữ liệu tương lai sau khối điền vào quá khứ $\implies$ Rò rỉ dữ liệu.<br>*Loại trừ xóa dòng:* Xóa dòng làm đứt gãy lưới thời gian liên tục. |
| Khuyết thiếu (Features)| Điền median trong Pipeline, fit trên Train | Median bền vững với phân bố lệch phải; mean bị kéo lệch bởi các đỉnh cao. |
| Ngoại lai | Giữ nguyên toàn bộ; sử dụng hàm mất mát MAE | Đỉnh tiêu thụ là hành vi thật cần dự báo; cắt tỉa sẽ làm sai tổng năng lượng và che giấu sai số đỉnh. |
| Rò rỉ dữ liệu | Loại bỏ `Global_intensity` và `Voltage` đồng thời | Tại thời điểm dự báo $t$, các thông số ở $t+h$ chưa hề xảy ra. |
| Mất cân bằng | Giữ nguyên chuỗi thời gian, không resample/SMOTE | SMOTE phá vỡ thứ tự chuỗi thời gian. Đánh giá riêng sai số theo nhóm đỉnh và khung giờ cao điểm. |

### 7.2 Thử nghiệm độ nhạy của ngưỡng số phút hợp lệ trong một giờ

**Bảng 7.2: Thử nghiệm các ngưỡng số phút có đo trong giờ**

| Ngưỡng tối thiểu (phút/giờ) | Số giờ hợp lệ | Tỷ lệ hợp lệ (%) | Số giờ bị loại (NaN) |
|:---:|:---:|:---:|:---:|
| 60 | 34.085 | 98,54% | 504 |
| **54 (Được chọn)** | **34.143** | **98,71%** | **446** |
| 30 | 34.157 | 98,75% | 432 |
| 1 | 34.168 | 98,78% | 421 |

*Nhận xét:* Thay đổi ngưỡng từ 60 xuống 1 phút chỉ làm thay đổi 83 giờ ($0,24\%$) vì phần lớn dữ liệu thiếu là các khối kéo dài nhiều ngày (trong đó 421 giờ mất trắng 100% không có phút nào, 25 giờ có 1–53 phút). Lựa chọn ngưỡng 54 phút cho phép thiếu tối đa 10% số phút đo, giữ lại được 58 giờ có mất dữ liệu vi mô nhưng giá trị trung bình vẫn đại diện chuẩn xác cho mức tiêu thụ của giờ đó.

### 7.3 Bảng so sánh trước (theo phút) và sau (theo giờ) xử lý

**Bảng 7.3: So sánh các chỉ số thống kê trước và sau tổng hợp theo giờ**

| Chỉ tiêu thống kê | Trước xử lý (GAP theo phút) | Sau xử lý (`energy_kwh` theo giờ) | Biến đổi vật lý & Nhận xét |
|---|:---:|:---:|---|
| Đơn vị đo lường | Công suất ($\text{kW}$) | Điện năng ($\text{kWh/giờ}$) | Quy đổi $E = P \times 1\text{h}$ |
| Số lượng dòng | 2.075.259 | 34.589 | Giảm 60 lần kích thước dữ liệu |
| Số dòng thiếu (NaN) | 25.979 ($1,252\%$) | 446 ($1,289\%$) | Tăng nhẹ do loại các giờ $< 54$ phút |
| Giá trị trung bình | **$1,0916\text{ kW}$** | **$1,0915\text{ kWh}$** | **Bảo toàn điện năng xấp xỉ** |
| Độ lệch chuẩn (Std) | $1,057\text{ kW}$ | $0,898\text{ kWh}$ | Giảm 15% (làm mượt dao động ngắn hạn) |
| Trung vị (Median) | $0,602\text{ kW}$ | $0,782\text{ kWh}$ | Tăng do loại bỏ các phút dao động nền |
| Giá trị nhỏ nhất (Min) | $0,076\text{ kW}$ | $0,124\text{ kWh}$ | Nâng nền tải trung bình giờ |
| Giá trị lớn nhất (Max) | $11,122\text{ kW}$ | $6,561\text{ kWh}$ | Làm mượt các đỉnh xung nhịp vài phút |
| Hệ số lệch Skewness | +1,79 | +1,30 | Đưa phân bố về gần đối xứng hơn |

---

## CHƯƠNG 8: CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH VÀ PHÒNG CHỐNG RÒ RỈ (TIÊU CHÍ 8)

### 8.1 Phân chia Train / Test theo thời gian và Kiểm định chéo TimeSeriesSplit
* **Phân chia Train / Test:** Căn cứ vào `target_time < 2010-01-01` làm tập Train (26.523 mẫu có nhãn) và `target_time >= 2010-01-01` làm tập Test (7.620 mẫu có nhãn). Tuyệt đối không xáo trộn ngẫu nhiên (`shuffle=False`).
* **Kiểm định chéo TimeSeriesSplit trên Train:** Áp dụng 5 fold với kích thước kiểm định $2.160\text{ giờ}$ (90 ngày) và khoảng đệm an toàn `gap = 168` giờ (1 tuần).

**Bảng 8.1: Kích thước các fold kiểm định chéo TimeSeriesSplit trên Train**

| Fold | Số mẫu Train | Khoảng thời gian Train | Số mẫu Validation | Khoảng thời gian Validation |
|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 15.702 | 2006-12-24 $\to$ 2008-10-18 | 2.160 | 2008-10-25 $\to$ 2009-01-23 |
| Fold 2 | 17.862 | 2006-12-24 $\to$ 2009-01-23 | 2.160 | 2009-01-30 $\to$ 2009-04-30 |
| Fold 3 | 20.022 | 2006-12-24 $\to$ 2009-04-30 | 2.160 | 2009-05-07 $\to$ 2009-08-05 |
| Fold 4 | 22.182 | 2006-12-24 $\to$ 2009-08-05 | 2.160 | 2009-08-12 $\to$ 2009-11-10 |
| Fold 5 | 24.342 | 2006-12-24 $\to$ 2009-11-10 | 2.160 | 2009-11-17 $\to$ 2009-12-31 |
| **Test** | **26.523 (Toàn bộ Train)** | **2006-12-24 $\to$ 2009-12-31** | **7.620** | **2010-01-01 $\to$ 2010-11-26** |

### 8.2 Đường ống tiền xử lý Scikit-learn Pipeline (Fit chỉ trên Train)
* Cấu trúc `ColumnTransformer`:
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` cho đặc trưng phân loại `hour`.
  - `make_pipeline(SimpleImputer(strategy='median'), StandardScaler())` cho các đặc trưng số học độ trễ (`lag_0`, `lag_1`, `lag_5`, `lag_23`, `lag_167`).
* Tham số học được từ Train: Giá trị median của `lag_0` là $0,7939\text{ kWh}$; trung bình chuẩn hóa $\mu = 1,1004\text{ kWh}$, độ lệch chuẩn $\sigma = 0,9043\text{ kWh}$.

### 8.3 Minh chứng thực nghiệm: Fit trên toàn bộ dữ liệu gây rò rỉ dữ liệu

**Bảng 8.2: So sánh tham số khi fit chuẩn mực trên Train vs fit sai phạm trên toàn bộ dữ liệu**

| Đặc trưng | Mean chuẩn (Chỉ Fit Train) | Mean rò rỉ (Fit Toàn bộ) | Độ lệch Mean | Std chuẩn (Chỉ Fit Train) | Std rò rỉ (Fit Toàn bộ) |
|---|:---:|:---:|:---:|:---:|:---:|
| `lag_0` | **1,1004** | **1,0915** | **-0,0089** | 0,9043 | 0,8981 |
| `lag_1` | 1,1004 | 1,0915 | -0,0089 | 0,9043 | 0,8981 |
| `lag_5` | 1,1004 | 1,0915 | -0,0089 | 0,9043 | 0,8981 |
| `lag_23` | 1,1004 | 1,0915 | -0,0089 | 0,9044 | 0,8981 |
| `lag_167` | 1,1001 | 1,0916 | -0,0085 | 0,9042 | 0,8981 |

*Kết luận:* Mặc dù sai lệch số học khoảng $-0,0089\text{ kWh}$, nhưng việc fit trên toàn bộ dữ liệu đã để lọt thông tin phân phối của năm 2010 vào tham số mô hình, vi phạm nguyên tắc nhân quả của chuỗi thời gian. Việc đóng gói toàn bộ quy trình vào Pipeline và chỉ gọi `.fit()` trên `X_train` là điều kiện tiên quyết để bảo vệ tính trung thực của kết quả học máy.

---

## CHƯƠNG 9: CHIẾN LƯỢC ĐÁNH GIÁ VÀ HỆ THỐNG CHỈ SỐ (TIÊU CHÍ 9)

1. **Vì sao chọn TimeSeriesSplit thay vì K-Fold ngẫu nhiên?**
   K-Fold ngẫu nhiên sẽ xáo trộn các mốc thời gian, khiến mô hình học các mẫu ở tương lai để dự báo quá khứ. Điều này vi phạm nghiêm trọng tính nhân quả thực tế. `TimeSeriesSplit` giữ vững trật tự dòng thời gian, mở rộng dần tập huấn luyện để mô phỏng chính xác kịch bản vận hành thực tế.
2. **Vì sao thiết lập `gap = 168` giờ?**
   Nhãn của mẫu huấn luyện cuối cùng không được phép chồng lấn lên tập kiểm định. Khoảng cách 168 giờ (1 tuần) bằng đúng độ trễ lớn nhất trong không gian đặc trưng (`lag_167`), ngăn chặn hoàn toàn nguy cơ rò rỉ thông tin qua các đặc trưng tuần hoàn.
3. **Vì sao chọn MAE làm chỉ số đo lường chính?**
   MAE cùng đơn vị với nhãn ($\text{kWh}$), có tính trực quan cao đối với người vận hành năng lượng. So với RMSE, MAE không phóng đại quá mức sai số của các đỉnh phụ tải thực tế, giúp đánh giá công bằng chất lượng dự báo tổng thể.

---

## CHƯƠNG 10: THIẾT LẬP THAO TÁC THỰC HÀNH TẠI CHỖ (TIÊU CHÍ 10)

Để sẵn sàng cho phần vấn đáp thực hành tại chỗ của Giảng viên (thao tác dưới 1 phút), nhóm đã thiết lập:
1. **Ô tham số tập trung ở đầu `02_chuan_bi_du_lieu.ipynb` (Ô 4):**
   ```python
   H = 1                         # Đổi tầm dự báo (1, 6, 12, 24)
   THRESH = 54                   # Đổi ngưỡng phút hợp lệ
   IMPUTE_STRATEGY = "median"    # Đổi chiến lược: "mean", "most_frequent"
   TEST_START = "2010-01-01"     # Đổi mốc phân chia thời gian
   STAT_COLUMN = "lag_0"         # Đổi cột khảo sát nhanh
   ```
2. **Hàm thao tác nhanh `thao_tac_doi_phuong_phap_dien_khuyet(new_strategy)` (Ô 19):**
   Cho phép chuyển đổi phương pháp điền khuyết sang `mean` chỉ với một dòng gọi hàm và in ra tham số học được ngay lập tức.
3. **Các hàm tiện ích trong `01_eda.ipynb` (Ô 2):**
   `thong_ke_cot(df, cot)` và `ve_phan_bo(df, cot, don_vi)` giúp thống kê và vẽ nhanh phân bố bất kỳ thuộc tính nào theo yêu cầu.

---

## PHỤ LỤC: HƯỚNG DẪN CHẠY LẠI VÀ PHẠM VI SỬ DỤNG CÔNG CỤ AI

### P.1 Hướng dẫn cài đặt và Chạy lại (Reproducibility)
1. Kích hoạt môi trường ảo:
   ```powershell
   cd cao_phim
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Chạy tuần tự các notebook:
   - Mở Jupyter Notebook / JupyterLab và chọn Kernel `.venv`.
   - Chạy `01_eda.ipynb`: Thực hiện toàn bộ quy trình EDA, định lượng chất lượng và xuất `hourly.parquet`.
   - Chạy `02_chuan_bi_du_lieu.ipynb`: Thực hiện tạo bảng đặc trưng, chia tập, kiểm định chéo và chạy Pipeline tiền xử lý.

### P.2 Khai báo phạm vi sử dụng công cụ AI (Academic Integrity)
Tuân thủ quy định học chính trực của môn học, nhóm công khai minh bạch:
* **Công cụ sử dụng:** Trợ lý AI lập trình (Antigravity AI Pair Programmer).
* **Phạm vi hỗ trợ:** Hỗ trợ rà soát công thức hiển thị LaTeX, hỗ trợ viết khung cấu trúc hàm tự động hóa việc tính toán các chỉ số định lượng theo tài liệu `docs/guide_bkt1_notebook.md`, kiểm tra lỗi cảnh báo cú pháp Pandas 3.1 (`day_of_week`), và định dạng bảng báo cáo Markdown.
* **Cam kết của sinh viên:** Toàn bộ logic tiền xử lý, quyết định lựa chọn ngưỡng kỹ thuật, phân tích nhận xét và các con số thực nghiệm đều được hai thành viên kiểm chứng thực tế và giải thích trọn vẹn trong buổi vấn đáp.
