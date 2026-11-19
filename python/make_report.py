"""Turn results/benchmark.csv into a Markdown table + bar chart for the README.

    python python/make_report.py --csv results/benchmark.csv
"""
import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/benchmark.csv")
    p.add_argument("--png", default="results/latency.png")
    p.add_argument("--md", default="results/benchmark.md")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["runtime", "threads"], keep="last")
    df = df.sort_values("infer_mean_ms")
    cols = ["runtime", "threads", "infer_mean_ms", "infer_p50_ms", "infer_p90_ms", "total_ms", "fps"]
    table = df[cols].to_markdown(index=False, floatfmt=".2f")
    with open(args.md, "w") as f:
        f.write(table + "\n")
    print(table)

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1.2))
    labels = [f"{r} (t={t})" if t else r for r, t in zip(df.runtime, df.threads)]
    ax.barh(labels, df.infer_mean_ms, color="#4C72B0")
    for y, v in enumerate(df.infer_mean_ms):
        ax.text(v, y, f" {v:.1f} ms", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("Inference latency, ms (mean, lower is better)")
    ax.set_title("YOLO11n @640, CPU")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print(f"Saved {args.png}, {args.md}")


if __name__ == "__main__":
    main()
