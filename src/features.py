"""Chuyển chuỗi điện năng theo giờ thành bảng hồi quy có giám sát cho một tầm dự báo h.

Quy ước (giống notebook 02):
- Mỗi dòng ứng với một thời điểm gốc t trên lưới giờ liên tục.
- Mọi đặc trưng lấy từ chuỗi y chỉ dùng giá trị tại các giờ <= t.
- Đặc trưng lịch mô tả thời điểm MỤC TIÊU t + h (biết trước).
- target = y(t + h), target_time = t + h.
Các dòng có NaN được giữ lại để lưới giờ liên tục; việc loại dòng thiếu nhãn để ở bước chia dữ liệu.
"""
from pathlib import Path

import pandas as pd

# lag_k = y(t - k). Chọn theo hàm tự tương quan trên tập huấn luyện (notebook 01, mục 8.2):
# các giờ gần nhất, nửa ngày, một ngày, hai ngày và một tuần trước.
LAGS = [0, 1, 2, 3, 5, 11, 23, 47, 167]
# Cửa sổ trượt kết thúc tại t (giống notebook 02, mục 11): 3 giờ, 1 ngày, 1 tuần.
ROLL_WINDOWS = [3, 24, 168]
ROLL_MIN_FRACTION = 0.75          # cửa sổ cần ít nhất int(0,75 * w) giờ có giá trị (giống notebook 02)
CALENDAR = ["hour", "dayofweek", "month", "is_weekend"]
CATEGORICAL = ["hour", "dayofweek", "month"]   # mã hóa one-hot cho mô hình tuyến tính
# is_weekend trùng thông tin với dayofweek đã mã hóa one-hot nên không đưa vào mô hình tuyến tính
LINEAR_EXCLUDE = ["is_weekend"]


def load_hourly(path="../data/processed/hourly.parquet") -> pd.Series:
    """Đọc chuỗi điện năng theo giờ (kWh) do notebook 01 tạo ra."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy chạy notebook 01_eda.ipynb trước.")
    return pd.read_parquet(path)["energy_kwh"]


def make_supervised(y: pd.Series, h: int) -> pd.DataFrame:
    """Tạo bảng (đặc trưng, nhãn) cho tầm dự báo h từ chuỗi giờ y."""
    if h < 1 or h > 24:
        raise ValueError("Đề tài dùng h trong khoảng 1..24 giờ")
    step = y.index.to_series().diff().dropna()
    if not step.eq(pd.Timedelta("1h")).all():
        raise ValueError("y phải nằm trên lưới giờ liên tục")

    cols = {f"lag_{k}": y.shift(k) for k in LAGS}
    # Giá trị cùng giờ với thời điểm mục tiêu ở ngày trước và tuần trước: y(t+h-24), y(t+h-168)
    cols["same_hour_prev_day"] = y.shift(24 - h)
    cols["same_hour_prev_week"] = y.shift(168 - h)
    for w in ROLL_WINDOWS:
        roll = y.rolling(w, min_periods=int(w * ROLL_MIN_FRACTION))   # cửa sổ (t-w, t]
        cols[f"roll_mean_{w}"] = roll.mean()
        cols[f"roll_std_{w}"] = roll.std()

    df = pd.DataFrame(cols, index=y.index)
    target_time = y.index + pd.Timedelta(hours=h)
    df["hour"] = target_time.hour
    df["dayofweek"] = target_time.dayofweek
    df["month"] = target_time.month
    df["is_weekend"] = (target_time.dayofweek >= 5).astype(int)
    df["target_time"] = target_time
    df["target"] = y.shift(-h)
    df.index.name = "origin"
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Các cột dùng làm đầu vào của mô hình (bỏ nhãn và thời điểm mục tiêu)."""
    return [c for c in df.columns if c not in ("target", "target_time")]


def train_test_tables(df: pd.DataFrame, test_start="2010-01-01"):
    """Chia theo thời điểm mục tiêu; chỉ giữ các dòng có nhãn. Không xáo trộn."""
    labeled = df[df["target"].notna()]
    is_train = labeled["target_time"] < pd.Timestamp(test_start)
    return labeled[is_train], labeled[~is_train]
