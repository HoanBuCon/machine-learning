"""Xuất các hình của notebook ra PDF (vector) cho báo cáo.

Chạy lại toàn bộ ô mã của từng notebook theo đúng thứ tự. Ô vẽ hình nào có dòng đầu
`# HINH: ten` thì hình được lưu thành bao_cao/figures/ten.pdf; các hình khác chỉ bị đóng lại.
Vì chạy lại đúng mã của notebook nên hình trong báo cáo khớp với notebook.

Cách chạy (từ thư mục cao_phim/notebooks):
    ../.venv/Scripts/python.exe ../bao_cao/scripts/xuat_hinh.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42          # nhúng font TrueType để giữ dấu tiếng Việt
OUT = Path(__file__).resolve().parents[1] / "figures"
NOTEBOOKS = ["01_eda.ipynb", "02_chuan_bi_du_lieu.ipynb", "03_huan_luyen_danh_gia.ipynb"]


def run(notebook: str) -> None:
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    ns = {"display": lambda *a, **k: None, "__name__": "__main__"}
    for idx, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        first = source.split("\n", 1)[0]
        name = first.split(":", 1)[1].strip() if first.startswith("# HINH:") else None

        def show(*_, _name=name):
            fig = plt.gcf()
            if _name:
                fig.savefig(OUT / f"{_name}.pdf", bbox_inches="tight")
                print("  đã lưu", f"{_name}.pdf")
            plt.close(fig)

        plt.show = show
        exec(compile(source, f"{notebook}[{idx}]", "exec"), ns)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for nb in (sys.argv[1:] or NOTEBOOKS):
        print(nb)
        run(nb)
