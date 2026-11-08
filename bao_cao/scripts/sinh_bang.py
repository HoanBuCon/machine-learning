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
        best = d["MAE_mean"].idxmin()
        for i, (m, r) in enumerate(d.iterrows()):
            mae = f"{num(r.MAE_mean)} $\\pm$ {num(r.MAE_std)}"
            if m == best:
                mae = f"\\textbf{{{mae}}}"
            hcol = f"\\multirow{{7}}{{*}}{{{h}}}" if i == 0 else ""
            rows.append(f"{hcol} & {m} & {mae} & {num(r.RMSE_mean)} $\\pm$ {num(r.RMSE_std)} & "
                        f"{num(r.MAPE_mean, 1)} $\\pm$ {num(r.MAPE_std, 1)} \\\\")
            rows.append("\\cline{2-5}" if i < len(d) - 1 else "\\hline")
    body = "\n".join(rows)
    write("tab_cv_summary.tex", rf"""\begin{{table}}[htbp]
\centering
\caption[Kết quả kiểm định chéo của mọi mô hình]{{Kết quả kiểm định chéo 5 fold trên tập huấn luyện (trung bình $\pm$ độ lệch chuẩn qua các fold). In đậm: MAE thấp nhất ở mỗi tầm dự báo.}}
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


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tab_cv_summary()
    tab_vs_baseline()
    tab_train_val()
    tab_error_conc()
