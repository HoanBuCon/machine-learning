"""Chỉ số đánh giá và quy trình kiểm định chéo theo thời gian."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models import BASELINE_INPUTS, BASELINES, MODELS, baseline_predict


def metrics(y_true, y_pred) -> dict:
    """MAE (chỉ số chính), RMSE và MAPE (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": np.mean(np.abs(err / y_true)) * 100,   # nhãn luôn > 0 (nhỏ nhất 0,124 kWh)
    }


def time_series_folds(train: pd.DataFrame, n_splits=5, test_size=2160, gap=168):
    """Các fold TimeSeriesSplit trên tập huấn luyện (đã sắp theo thời gian, chỉ gồm dòng có nhãn)."""
    return list(TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap).split(train))


def run_cv(train: pd.DataFrame, feature_cols: list[str], folds, models=None):
    """Kiểm định chéo cho mọi mô hình cơ sở và mô hình học máy trên cùng các fold.

    Trả về:
    - scores: một dòng cho mỗi (fold, mô hình, tập) với tập = "train" hoặc "val";
    - oof: dự báo trên phần kiểm định của mọi fold (để phân tích lỗi);
    - fitted: mô hình đã huấn luyện ở fold cuối (để tính tầm quan trọng đặc trưng).
    """
    models = models or MODELS
    rows, oof, fitted = [], [], {}
    for k, (tr_idx, val_idx) in enumerate(folds, start=1):
        tr, val = train.iloc[tr_idx], train.iloc[val_idx]
        # Cùng tập mẫu kiểm định cho mọi mô hình: bỏ các giờ mà mô hình cơ sở thiếu giá trị quá khứ
        val = val[val[BASELINE_INPUTS].notna().all(axis=1)]
        pred = pd.DataFrame({"target": val["target"], "target_time": val["target_time"], "fold": k},
                            index=val.index)

        for name in BASELINES:
            pred[name] = baseline_predict(name, tr, val)
            rows.append({"fold": k, "model": name, "set": "val", **metrics(val["target"], pred[name])})

        for name, make in models.items():
            model = make(feature_cols)
            model.fit(tr[feature_cols], tr["target"])
            pred[name] = model.predict(val[feature_cols])
            rows.append({"fold": k, "model": name, "set": "val", **metrics(val["target"], pred[name])})
            rows.append({"fold": k, "model": name, "set": "train",
                         **metrics(tr["target"], model.predict(tr[feature_cols]))})
            if k == len(folds):
                fitted[name] = (model, val)
        oof.append(pred)
    return pd.DataFrame(rows), pd.concat(oof), fitted

