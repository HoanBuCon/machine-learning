"""Sinh các bảng kết quả LaTeX (bao_cao/tables/*.tex) từ thư mục ket_qua/ do notebook 03 ghi ra.

Cách chạy (từ thư mục cao_phim):
    .venv/Scripts/python.exe bao_cao/scripts/sinh_bang.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "ket_qua", ROOT / "bao_cao" / "tables"
ORDER = ["Mean", "Naive", "SNaive24", "SNaive168", "LinearRegression", "RandomForest", "XGBoost"]
ML = ["LinearRegression", "RandomForest", "XGBoost"]


def num(x, d=3):
    """Số kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{x:.{d}f}".replace(".", "{,}").replace("-", "$-$")


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print("đã ghi", name)


def tab_cv_summary():
    s = pd.read_csv(RES / "cv_summary.csv")
    rows = []
    for h in sorted(s.h.unique()):
        d = s[s.h == h].set_index("model").loc[ORDER]
        for i, (m, r) in enumerate(d.iterrows()):
            mae = f"{num(r.MAE_mean)} $\\pm$ {num(r.MAE_std)}"
            hcol = f"\\multirow{{7}}{{*}}{{{h}}}" if i == 0 else ""
            rows.append(f"{hcol} & {m} & {mae} & {num(r.RMSE_mean)} $\\pm$ {num(r.RMSE_std)} & "
                        f"{num(r.MAPE_mean, 1)} $\\pm$ {num(r.MAPE_std, 1)} \\\\")
            rows.append("\\cline{2-5}" if i < len(d) - 1 else "\\hline")
    body = "\n".join(rows)
    write("tab_cv_summary.tex", rf"""\begin{{table}}[htbp]
\centering
\caption[Kết quả kiểm định chéo của mọi mô hình]{{Kết quả kiểm định chéo 5 fold trên tập huấn luyện (trung bình $\pm$ độ lệch chuẩn qua các fold).}}
\label{{tab:cv_summary}}
\small
\begin{{tabular}}{{|c|l|c|c|c|}}
\hline
\textbf{{$h$}} & \textbf{{Mô hình}} & \textbf{{MAE (kWh)}} & \textbf{{RMSE (kWh)}} & \textbf{{MAPE (\%)}} \\
\hline
{body}
\end{{tabular}}
\end{{table}}
""")


def tab_vs_baseline():
    c = pd.read_csv(RES / "cv_vs_baseline.csv")
    rows = []
    for h in sorted(c.h.unique()):
        d = c[c.h == h]
        for i, (_, r) in enumerate(d.iterrows()):
            first = f"\\multirow{{3}}{{*}}{{{h}}} & \\multirow{{3}}{{*}}{{{r['Baseline tốt nhất']} ({num(r['MAE baseline'])})}}" if i == 0 else " & "
            rows.append(f"{first} & {r['Mô hình']} & {num(r['MAE mô hình'])} & {num(r['Giảm MAE (%)'], 1)} & {int(r['Số fold tốt hơn'])}/5 \\\\")
            rows.append("\\cline{3-6}" if i < len(d) - 1 else "\\hline")
    body = "\n".join(rows)
    write("tab_vs_baseline.tex", rf"""\begin{{table}}[htbp]
\centering
\caption[So sánh mô hình học máy với mô hình cơ sở tốt nhất]{{So sánh MAE trung bình của mô hình học máy với mô hình cơ sở tốt nhất ở từng tầm dự báo; cột cuối là số fold mô hình học máy có MAE thấp hơn.}}
\label{{tab:vs_baseline}}
\small
\begin{{tabular}}{{|c|c|l|c|c|c|}}
\hline
\textbf{{$h$}} & \textbf{{Cơ sở tốt nhất (MAE)}} & \textbf{{Mô hình}} & \textbf{{MAE}} & \textbf{{Giảm MAE (\%)}} & \textbf{{Số fold tốt hơn}} \\
\hline
{body}
\end{{tabular}}
\end{{table}}
""")


def tab_train_val():
    t = pd.read_csv(RES / "cv_train_vs_val.csv")
    rows = []
    for h in sorted(t.h.unique()):
        d = t[t.h == h].set_index("model").loc[ML]
        cells = " & ".join(f"{num(r['MAE huấn luyện'])} & {num(r['MAE kiểm định'])} & {num(r['Kiểm định / Huấn luyện'], 2)}"
                           for _, r in d.iterrows())
        rows.append(f"{h} & {cells} \\\\ \\hline")
    body = "\n".join(rows)
    write("tab_train_val.tex", rf"""\begin{{table}}[htbp]
\centering
\caption[MAE trên phần huấn luyện và phần kiểm định]{{MAE trung bình qua 5 fold trên phần huấn luyện (trong mẫu, ký hiệu HL) và phần kiểm định (KĐ), cùng tỷ số KĐ/HL.}}
\label{{tab:train_val}}
\small
\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|}}
\hline
\multirow{{2}}{{*}}{{\textbf{{$h$}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{Linear Regression}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{Random Forest}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{XGBoost}}}} \\
\cline{{2-10}}
 & HL & KĐ & KĐ/HL & HL & KĐ & KĐ/HL & HL & KĐ & KĐ/HL \\
\hline
{body}
\end{{tabular}}
\end{{table}}
""")


def tab_error_conc():
    e = pd.read_csv(RES / "cv_error_concentration.csv")
    rows = []
    for _, r in e.iterrows():
        rows.append(f"{int(r.h)} & {r['Mô hình']} & {num(r['Sai lệch TB ở 5% giờ cao nhất (kWh)'])} & "
                    f"{num(r['Sai lệch TB ở 10% giờ thấp nhất (kWh)'])} & "
                    f"{num(r['Tỷ trọng sai số của 10% giờ sai nhiều nhất (%)'], 1)} \\\\ \\hline")
    body = "\n".join(rows)
    write("tab_error_conc.tex", rf"""\begin{{table}}[htbp]
\centering
\caption[Mức độ tập trung của sai số]{{Sai lệch trung bình (dự báo $-$ thực tế) ở nhóm giờ tiêu thụ cao nhất và thấp nhất, và tỷ trọng tổng sai số tuyệt đối do 10\% số giờ có sai số lớn nhất gây ra (dự báo kiểm định của 5 fold).}}
\label{{tab:error_conc}}
\small
\begin{{tabular}}{{|c|l|c|c|c|}}
\hline
\textbf{{$h$}} & \textbf{{Mô hình}} & \textbf{{5\% giờ cao nhất}} & \textbf{{10\% giờ thấp nhất}} & \textbf{{Tỷ trọng (\%)}} \\
\hline
{body}
\end{{tabular}}
\end{{table}}
""")


# ------------------------------------------------------------------ bảng của notebook 04
HS = [1, 6, 12, 24]
NAMES = {"LinearRegression": "Linear Regression", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
         "Ridge": "Ridge", "Voting": "Voting (RF + XGBoost)"}


def table(name, caption, short, label, colspec, header, body, size=r"\small"):
    write(name, rf"""\begin{{table}}[htbp]
\centering
\caption[{short}]{{{caption}}}
\label{{{label}}}
{size}
\begin{{tabular}}{{{colspec}}}
\hline
{header} \\
\hline
{body}
\end{{tabular}}
\end{{table}}
""")


def tab_feat_delta():
    d = pd.read_csv(RES / "toi_uu_feature_delta.csv")
    sets = {"SUB": "Công tơ phụ", "HOL": "Ngày lễ", "ALL": "Cả hai"}
    rows = []
    for mi, m in enumerate(ML):
        for si, (fs, label) in enumerate(sets.items()):
            s = d[(d.model == m) & (d.features == fs)].set_index("h")
            cells = " & ".join(f"{num(s.loc[h, 'Thay đổi (%)'], 1)} ({int(s.loc[h, 'Số fold tốt hơn BASE'])}/5)" for h in HS)
            mcol = f"\\multirow{{3}}{{*}}{{{NAMES[m]}}}" if si == 0 else ""
            rows.append(f"{mcol} & {label} & {cells} \\\\")
            rows.append("\\cline{2-6}" if si < 2 else "\\hline")
    table("tab_feat_delta.tex",
          "Thay đổi MAE kiểm định chéo (\\%) khi thêm đặc trưng mới so với bộ 21 đặc trưng ban đầu, siêu tham số giữ nguyên; trong ngoặc là số fold có MAE thấp hơn. Giá trị âm là tốt hơn.",
          "Tác động của đặc trưng mới", "tab:feat_delta", "|l|l|c|c|c|c|",
          "\\textbf{Mô hình} & \\textbf{Đặc trưng thêm} & \\textbf{$h = 1$} & \\textbf{$h = 6$} & \\textbf{$h = 12$} & \\textbf{$h = 24$}",
          "\n".join(rows))


def tab_best_params():
    p = pd.read_csv(RES / "toi_uu_best_params.csv")
    cols = {"RandomForest": [("min_samples_leaf", "Số mẫu tối thiểu ở lá"), ("max_features", "Tỷ lệ đặc trưng mỗi lần tách"),
                             ("max_depth", "Độ sâu tối đa")],
            "XGBoost": [("n_estimators", "Số cây"), ("learning_rate", "Tốc độ học"), ("max_depth", "Độ sâu tối đa"),
                        ("min_child_weight", "Tổng trọng số tối thiểu ở lá"), ("subsample", "Tỷ lệ mẫu mỗi cây"),
                        ("colsample_bytree", "Tỷ lệ đặc trưng mỗi cây"), ("reg_lambda", "Hệ số phạt L2 ($\\lambda$)")],
            "Ridge": [("alpha", "$\\alpha$")]}
    rows = []
    for m, params in cols.items():
        for i, (c, label) in enumerate(params):
            s = p[p["Mô hình"] == m].set_index("h")
            vals = []
            for h in HS:
                v = s.loc[h, c]
                if c == "max_features" or c in ("learning_rate", "subsample", "colsample_bytree"):
                    vals.append(num(v, 2))
                elif c == "alpha" or c == "reg_lambda":
                    vals.append(num(v, 2) if v < 1 else str(int(v)))
                else:
                    vals.append("không giới hạn" if pd.isna(v) else str(int(v)))
            mcol = f"\\multirow{{{len(params)}}}{{*}}{{{NAMES[m]}}}" if i == 0 else ""
            rows.append(f"{mcol} & {label} & {' & '.join(vals)} \\\\")
            rows.append(f"\\cline{{2-6}}" if i < len(params) - 1 else "\\hline")
    table("tab_best_params.tex", "Siêu tham số tốt nhất tìm được bằng tìm kiếm trên 5 fold kiểm định chéo.",
          "Siêu tham số tối ưu", "tab:best_params", "|l|l|c|c|c|c|",
          "\\textbf{Mô hình} & \\textbf{Siêu tham số} & \\textbf{$h = 1$} & \\textbf{$h = 6$} & \\textbf{$h = 12$} & \\textbf{$h = 24$}",
          "\n".join(rows), size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


def tab_ablation():
    a = pd.read_csv(RES / "toi_uu_ablation.csv")
    steps = [("Mô hình cơ sở", ["Naive", "SNaive168"]),
             ("A0. Bài kiểm tra 2", ML), ("A1. + đặc trưng mới", ML),
             ("A2. + tối ưu siêu tham số", ["Ridge", "RandomForest", "XGBoost"]), ("A3. + tổ hợp Voting", ["Voting"])]
    labels = {"Mô hình cơ sở": "Cơ sở", "A0. Bài kiểm tra 2": "A0", "A1. + đặc trưng mới": "A1",
              "A2. + tối ưu siêu tham số": "A2", "A3. + tổ hợp Voting": "A3"}
    best = {h: a[(a.h == h) & (a.step != "Mô hình cơ sở")].MAE_mean.min() for h in HS}
    rows = []
    for step, models in steps:
        for i, m in enumerate(models):
            cells = []
            for h in HS:
                r = a[(a.h == h) & (a.step == step) & (a.model == m)].iloc[0]
                cells.append(f"{num(r.MAE_mean)} $\\pm$ {num(r.MAE_std)}")
            scol = f"\\multirow{{{len(models)}}}{{*}}{{{labels[step]}}}" if i == 0 else ""
            rows.append(f"{scol} & {NAMES.get(m, m)} & {' & '.join(cells)} \\\\")
            rows.append("\\cline{2-6}" if i < len(models) - 1 else "\\hline")
    table("tab_ablation.tex",
          "MAE kiểm định chéo (kWh, trung bình $\\pm$ độ lệch chuẩn qua 5 fold) qua từng bước cải tiến, cùng dữ liệu, cùng 5 fold và cùng tập mẫu. A0: cấu hình Bài kiểm tra 2; A1: thêm đặc trưng công tơ phụ; A2: tối ưu siêu tham số; A3: tổ hợp Voting.",
          "So sánh trước và sau cải tiến (ablation)", "tab:ablation", "|c|l|c|c|c|c|",
          "\\textbf{Bước} & \\textbf{Mô hình} & \\textbf{$h = 1$} & \\textbf{$h = 6$} & \\textbf{$h = 12$} & \\textbf{$h = 24$}",
          "\n".join(rows), size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


def tab_test():
    t = pd.read_csv(RES / "test_metrics.csv")
    order = [("Mean", "Mean"), ("Naive", "Naive"), ("SNaive24", "SNaive24"), ("SNaive168", "SNaive168"),
             ("A0 LinearRegression", "Linear Regression (A0)"), ("A0 RandomForest", "Random Forest (A0)"),
             ("A0 XGBoost", "XGBoost (A0)"), ("A2/A3 Ridge", "Ridge (A2)"), ("A2/A3 RandomForest", "Random Forest (A2)"),
             ("A2/A3 XGBoost", "XGBoost (A2, mô hình cuối)"), ("A2/A3 Voting", "Voting (A3)")]
    rows = []
    for key, label in order:
        s = t[t.model == key].set_index("h")
        rows.append(f"{label} & " + " & ".join(num(s.loc[h, 'MAE']) for h in HS) + " \\\\ \\hline")
    table("tab_test.tex",
          "MAE (kWh) trên tập kiểm tra năm 2010 (7.227 giờ cho mỗi tầm dự báo). Mô hình được huấn luyện lại trên toàn bộ tập huấn luyện với cấu hình đã chốt.",
          "Kết quả trên tập kiểm tra", "tab:test", "|l|c|c|c|c|",
          "\\textbf{Mô hình} & \\textbf{$h = 1$} & \\textbf{$h = 6$} & \\textbf{$h = 12$} & \\textbf{$h = 24$}",
          "\n".join(rows))


def tab_cv_test():
    c = pd.read_csv(RES / "test_cv_vs_test.csv").set_index("h")
    rows = []
    for h in HS:
        r = c.loc[h]
        rows.append(f"{h} & {num(r['MAE kiểm định chéo'])} & {num(r['MAE kiểm tra'])} & {num(r['RMSE kiểm tra'])} & "
                    f"{num(r['MAPE kiểm tra'], 1)} & {r['Baseline chọn trên CV']} ({num(r['MAE baseline đó (kiểm tra)'])}) & "
                    f"{num(r['Giảm MAE so với baseline chọn trên CV (%)'], 1)} \\\\ \\hline")
    table("tab_cv_test.tex",
          "Mô hình cuối (XGBoost đã tối ưu): MAE kiểm định chéo so với kết quả trên tập kiểm tra, và mức giảm MAE so với mô hình cơ sở được chọn trên kiểm định chéo (MAE của mô hình cơ sở đó trên tập kiểm tra trong ngoặc).",
          "Kiểm định chéo và tập kiểm tra của mô hình cuối", "tab:cv_test", "|c|c|c|c|c|c|c|",
          "\\textbf{$h$} & \\textbf{MAE KĐC} & \\textbf{MAE KT} & \\textbf{RMSE KT} & \\textbf{MAPE KT (\\%)} & \\textbf{Cơ sở (MAE KT)} & \\textbf{Giảm (\\%)}",
          "\n".join(rows), size=r"\footnotesize\setlength{\tabcolsep}{4pt}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tab_cv_summary()
    tab_vs_baseline()
    tab_train_val()
    tab_error_conc()
    tab_feat_delta()
    tab_best_params()
    tab_ablation()
    tab_test()
    tab_cv_test()
