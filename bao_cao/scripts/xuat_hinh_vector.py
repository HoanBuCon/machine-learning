"""Xuất lại hình của hai notebook dưới dạng vector (PDF) cho báo cáo.

Chạy toàn bộ ô mã của notebook theo đúng thứ tự, thay plt.show() bằng lưu PDF,
nên hình trong báo cáo khớp với notebook. Cách chạy (từ thư mục cao_phim/notebooks):
    ../.venv/Scripts/python.exe ../bao_cao/scripts/xuat_hinh_vector.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42  # nhúng font TrueType để giữ dấu tiếng Việt
OUT = Path(__file__).resolve().parents[1] / "tai_nguyen" / "hinh"


def chay(notebook, tag):
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    ns = {"display": lambda *a, **k: None}
    for idx, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue

        def show(*_, _idx=idx):
            fig = plt.gcf()
            if (tag, _idx) == ("01", 33):  # nhãn ngày ngắn dd/mm để không chồng nhau
                ax = fig.axes[1]
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
                ax.set_xlabel("Thời gian (ngày/tháng, năm 2010)")
            fig.savefig(OUT / f"hinh-{tag}-c{_idx:02d}.pdf", bbox_inches="tight")
            print("  saved", f"hinh-{tag}-c{_idx:02d}.pdf")
            plt.close(fig)

        plt.show = show
        exec(compile("".join(cell["source"]), f"{tag}[{idx}]", "exec"), ns)


chay("01_eda.ipynb", "01")
chay("02_chuan_bi_du_lieu.ipynb", "02")
