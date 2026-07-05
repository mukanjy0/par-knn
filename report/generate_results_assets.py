import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
GEN = REPORT / "generated"
MPL_CACHE = REPORT / ".matplotlib"
MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib.pyplot as plt

PATHS = {
    "v2_strong": ROOT / "khipu_v2_results/khipu_v2_strong_20260705_145227_job44803/summary.csv",
    "v2_weak": ROOT / "khipu_v2_results/khipu_v2_weak_20260705_145816_job44804/summary.csv",
    "v2_accuracy": ROOT / "khipu_v2_results/khipu_v2_accuracy_20260705_145910_job44805/summary.csv",
    "v3_strong": ROOT / "khipu_v3_results/khipu_v3_strong_20260705_152151_job44813/summary.csv",
    "v3_weak": ROOT / "khipu_v3_results/khipu_v3_weak_20260705_153745_job44814/summary.csv",
    "v3_accuracy": ROOT / "khipu_v3_results/khipu_v3_accuracy_20260705_153947_job44815/summary.csv",
}


def read(name):
    df = pd.read_csv(PATHS[name])
    df["version"] = name.split("_")[0].upper()
    return df


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def latex_table(path, tabular, caption, label):
    body = "\\begin{table}[H]\n\\centering\n"
    body += tabular
    body += f"\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    path.write_text(body, encoding="utf-8")


def make_plots(strong, weak, accuracy):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"V2": "#1f77b4", "V3": "#d62728"}

    largest = 100000
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for version, df in strong.groupby("version"):
        part = df[df.n_train == largest].sort_values("p")
        ax.plot(part.p, part.speedup, marker="o", label=version, color=colors[version])
    ax.plot([1, 32], [1, 32], linestyle="--", color="0.4", label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Procesos MPI (p)")
    ax.set_ylabel("Speedup")
    ax.set_title("Strong scaling, n_train=100000")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "strong_speedup_ntrain100000.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for version, df in strong.groupby("version"):
        part = df[df.n_train == largest].sort_values("p")
        ax.plot(part.p, part.t_total_median, marker="o", label=version, color=colors[version])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Procesos MPI (p)")
    ax.set_ylabel("Tiempo total mediano (s)")
    ax.set_title("Tiempo total, n_train=100000")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "strong_time_ntrain100000.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for version, df in strong.groupby("version"):
        part = df[df.n_train == largest].sort_values("p")
        ax.plot(part.p, part.flops_per_sec_median / 1e9, marker="o", label=version, color=colors[version])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Procesos MPI (p)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Rendimiento estimado, n_train=100000")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "strong_gflops_ntrain100000.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for version, df in weak.groupby("version"):
        part = df.sort_values("p")
        ax.plot(part.p, part.t_total_median, marker="o", label=version, color=colors[version])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Procesos MPI (p)")
    ax.set_ylabel("Tiempo total mediano (s)")
    ax.set_title("Weak scaling, 3125 train/proceso")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "weak_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for version, df in accuracy.groupby("version"):
        part = df[df.p == 1].sort_values("n")
        ax.plot(part.n, part.accuracy_mean, marker="o", label=version, color=colors[version])
    ax.set_xscale("log")
    ax.set_xticks([10000, 50000, 100000])
    ax.set_xticklabels(["10000", "50000", "100000"])
    ax.set_ylim(0.98, 0.992)
    ax.set_xlabel("n total")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy con split 80/20")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "accuracy.png", dpi=180)
    plt.close(fig)


def make_tables(strong, weak, accuracy):
    rows = []
    for version, df in strong.groupby("version"):
        for n_train, part in df.groupby("n_train"):
            part = part.sort_values("t_total_median")
            best = part.iloc[0]
            p32 = df[(df.n_train == n_train) & (df.p == 32)].iloc[0]
            rows.append([
                version,
                str(int(n_train)),
                str(int(best.p)),
                fmt(best.t_total_median, 3),
                fmt(best.speedup, 2),
                fmt(p32.speedup, 2),
                fmt(p32.efficiency, 2),
            ])
    table = "\\begin{tabular}{llrrrrr}\n\\toprule\nVersión & $n_{train}$ & Mejor $p$ & $T_{min}$ (s) & $S(T_{min})$ & $S_{32}$ & $E_{32}$ \\\\\n\\midrule\n"
    for r in rows:
        table += " & ".join(r) + " \\\\\n"
    table += "\\bottomrule\n\\end{tabular}"
    latex_table(GEN / "strong_summary_table.tex", table,
                "Resumen de escalabilidad fuerte. El baseline de speedup es p=1 de la misma versión y el mismo tamaño.",
                "tab:strong-summary")

    part_rows = []
    for p in [1, 2, 4, 8, 16, 32]:
        row = [str(p)]
        for version in ["V2", "V3"]:
            item = weak[(weak.version == version) & (weak.p == p)].iloc[0]
            row += [str(int(item.n_train)), fmt(item.t_total_median, 3), fmt(item.flops_per_sec_median / 1e9, 1)]
        part_rows.append(row)
    table = "\\begin{tabular}{rrrrrrr}\n\\toprule\n$p$ & V2 $n_{train}$ & V2 $T$ (s) & V2 GF/s & V3 $n_{train}$ & V3 $T$ (s) & V3 GF/s \\\\\n\\midrule\n"
    for r in part_rows:
        table += " & ".join(r) + " \\\\\n"
    table += "\\bottomrule\n\\end{tabular}"
    latex_table(GEN / "weak_table.tex", table,
                "Escalabilidad débil con 3125 puntos de entrenamiento por proceso, $n_{test}=5000$, $k=3$.",
                "tab:weak")

    part_rows = []
    for n in [10000, 50000, 100000]:
        row = [str(n)]
        for version in ["V2", "V3"]:
            item = accuracy[(accuracy.version == version) & (accuracy.n == n) & (accuracy.p == 1)].iloc[0]
            p32 = accuracy[(accuracy.version == version) & (accuracy.n == n) & (accuracy.p == 32)].iloc[0]
            row += [fmt(item.accuracy_mean, 5), fmt(p32.speedup, 2)]
        part_rows.append(row)
    table = "\\begin{tabular}{rrrrr}\n\\toprule\n$n$ total & V2 accuracy & V2 $S_{32}$ & V3 accuracy & V3 $S_{32}$ \\\\\n\\midrule\n"
    for r in part_rows:
        table += " & ".join(r) + " \\\\\n"
    table += "\\bottomrule\n\\end{tabular}"
    latex_table(GEN / "accuracy_table.tex", table,
                "Accuracy y speedup de la prueba de correctitud. Las predicciones guardadas fueron idénticas entre p=1, 8 y 32.",
                "tab:accuracy")

    rows = []
    for p in [1, 2, 4, 8, 16, 32]:
        row = [str(p)]
        for version in ["V2", "V3"]:
            item = strong[(strong.version == version) & (strong.n_train == 100000) & (strong.p == p)].iloc[0]
            row += [fmt(item.t_total_median, 3), fmt(item.speedup, 2), fmt(item.efficiency, 2)]
        rows.append(row)
    table = "\\begin{tabular}{rrrrrrr}\n\\toprule\n$p$ & V2 $T$ & V2 $S$ & V2 $E$ & V3 $T$ & V3 $S$ & V3 $E$ \\\\\n\\midrule\n"
    for r in rows:
        table += " & ".join(r) + " \\\\\n"
    table += "\\bottomrule\n\\end{tabular}"
    latex_table(GEN / "strong_100k_table.tex", table,
                "Strong scaling para $n_{train}=100000$ y $n_{test}=5000$.",
                "tab:strong-100k")


def main():
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)

    strong = pd.concat([read("v2_strong"), read("v3_strong")], ignore_index=True)
    weak = pd.concat([read("v2_weak"), read("v3_weak")], ignore_index=True)
    accuracy = pd.concat([read("v2_accuracy"), read("v3_accuracy")], ignore_index=True)

    make_plots(strong, weak, accuracy)
    make_tables(strong, weak, accuracy)


if __name__ == "__main__":
    main()
