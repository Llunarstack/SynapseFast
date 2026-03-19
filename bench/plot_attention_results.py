from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--kind", choices=["bar", "line"], default="bar")
    args = p.parse_args()

    data = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    rows = data["rows"]
    meta = data.get("meta", {})

    Ts = [r["T"] for r in rows]
    series = {
        "torch": [r["torch_ms"] for r in rows],
        "sf_auto": [r["sf_auto_ms"] for r in rows],
        "sf_torch": [r["sf_torch_ms"] for r in rows],
        "sf_cuda": [r["sf_cuda_ms"] for r in rows],
    }
    if any(r.get("xformers_ms") is not None for r in rows):
        series["xformers"] = [r.get("xformers_ms") for r in rows]

    plt.figure(figsize=(10, 5))
    if args.kind == "line":
        for name, ys in series.items():
            if ys is None:
                continue
            if any(v is None for v in ys):
                continue
            plt.plot(Ts, ys, marker="o", label=name)
    else:
        names = [k for k, ys in series.items() if ys is not None and not any(v is None for v in ys)]
        x = np.arange(len(Ts), dtype=np.float32)
        width = 0.85 / max(1, len(names))
        offset0 = -0.5 * (len(names) - 1) * width
        for i, name in enumerate(names):
            ys = series[name]
            plt.bar(x + offset0 + i * width, ys, width=width, label=name)
        plt.xticks(x, Ts)

    title = f"Attention latency (ms) dtype={meta.get('dtype')} causal={meta.get('causal')} B={meta.get('B')} H={meta.get('H')} D={meta.get('D')}"
    plt.title(title)
    plt.xlabel("T (sequence length)")
    plt.ylabel("ms / call")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
