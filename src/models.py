"""Mô hình cơ sở và mô hình học máy.

- Mô hình cơ sở không học tham số từ đặc trưng (trừ "Mean" dùng trung bình nhãn của tập huấn luyện).
- Mỗi mô hình học máy là một Pipeline: mọi bước học tham số (điền khuyết, chuẩn hóa, one-hot)
  chỉ được fit trên phần huấn luyện của từng fold, nên không rò rỉ sang phần kiểm định.
- Chiến lược Direct: mỗi tầm dự báo h huấn luyện một mô hình riêng.
- Siêu tham số ở Tuần 6 được cố định trước (không dò tìm); tối ưu siêu tham số thuộc giai đoạn Tuần 8.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features import CATEGORICAL, LINEAR_EXCLUDE

RANDOM_STATE = 42

# Siêu tham số cố định cho Tuần 6 (giá trị phổ biến, không chọn theo kết quả kiểm định)
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, max_features=0.5)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8)


# ------------------------------------------------------------------ mô hình cơ sở
def baseline_predict(name: str, train: pd.DataFrame, val: pd.DataFrame) -> pd.Series:
    """Dự báo của mô hình cơ sở trên phần kiểm định."""
    if name == "Mean":          # trung bình nhãn của phần huấn luyện
        return pd.Series(train["target"].mean(), index=val.index)
    if name == "Naive":         # y(t+h) ≈ y(t)
        return val["lag_0"]
    if name == "SNaive24":      # cùng giờ ngày trước: y(t+h-24)
        return val["same_hour_prev_day"]
    if name == "SNaive168":     # cùng giờ tuần trước: y(t+h-168)
        return val["same_hour_prev_week"]
    raise ValueError(name)


BASELINES = ["Mean", "Naive", "SNaive24", "SNaive168"]
# Mẫu kiểm định chỉ được tính khi mọi mô hình cơ sở đều có giá trị, để mọi mô hình so sánh trên cùng tập mẫu
BASELINE_INPUTS = ["lag_0", "same_hour_prev_day", "same_hour_prev_week"]


# ------------------------------------------------------------------ mô hình học máy
def make_linear(feature_cols: list[str], **params) -> Pipeline:
    """Hồi quy tuyến tính: one-hot cho giờ/thứ/tháng, điền khuyết trung vị và chuẩn hóa phần còn lại.

    drop="first" bỏ một cột của mỗi nhóm one-hot để ma trận đặc trưng không bị suy biến cùng hệ số chặn,
    nhờ đó hệ số của mô hình được xác định duy nhất và diễn giải được.
    """
    numeric = [c for c in feature_cols if c not in CATEGORICAL + LINEAR_EXCLUDE]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ])
    return Pipeline([("pre", pre), ("model", LinearRegression(**params))])


def make_random_forest(feature_cols: list[str], **params) -> Pipeline:
    """Random Forest: cây không cần chuẩn hóa; điền khuyết trung vị giống mô hình tuyến tính."""
    rf = RandomForestRegressor(**{**RF_PARAMS, **params}, n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", rf)])


def make_xgboost(feature_cols: list[str], **params) -> XGBRegressor:
    """XGBoost tự xử lý giá trị thiếu khi tách nút nên không cần điền khuyết."""
    return XGBRegressor(**{**XGB_PARAMS, **params}, tree_method="hist", n_jobs=-1,
                        random_state=RANDOM_STATE)


MODELS = {
    "LinearRegression": make_linear,
    "RandomForest": make_random_forest,
    "XGBoost": make_xgboost,
}
