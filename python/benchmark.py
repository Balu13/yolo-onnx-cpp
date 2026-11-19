"""Python baselines for the speed comparison table.

Measures, on the same image and with the same warm-up/iterations as the C++ app:
  * py-pytorch      : Ultralytics model on PyTorch (.pt)
  * py-onnxruntime  : ONNX Runtime Python API (inference only, like the C++ timing)

    python python/benchmark.py --pt runs/.../best.pt --onnx models/best.onnx \
        --image assets/test.jpg --csv results/benchmark.csv
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"mean": a.mean(), "p50": np.percentile(a, 50),
            "p90": np.percentile(a, 90), "p99": np.percentile(a, 99)}


def bench_ultralytics(pt: str, image: str, imgsz: int, warmup: int, iters: int) -> dict:
    model = YOLO(pt)
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
    pre, inf, post = [], [], []
    for _ in range(iters):
        r = model.predict(image, imgsz=imgsz, device="cpu", verbose=False)[0]
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    return {"pre": np.mean(pre), "inf": stats(inf), "post": np.mean(post)}


def bench_ort(onnx_path: str, imgsz: int, warmup: int, iters: int, threads: int) -> dict:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    inf = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        inf.append((time.perf_counter() - t0) * 1000)
    return {"pre": float("nan"), "inf": stats(inf), "post": float("nan")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--csv", default="results/benchmark.csv")
    args = p.parse_args()

    rows = {
        "py-pytorch": bench_ultralytics(args.pt, args.image, args.imgsz, args.warmup, args.iters),
        "py-onnxruntime": bench_ultralytics(args.onnx, args.image, args.imgsz, args.warmup, args.iters),
    }

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["runtime", "model", "imgsz", "threads", "pre_ms", "infer_mean_ms",
                        "infer_p50_ms", "infer_p90_ms", "infer_p99_ms", "post_ms", "total_ms", "fps"])
        for rt, r in rows.items():
            i = r["inf"]
            total = np.nansum([r["pre"], i["mean"], r["post"]])
            model = args.pt if rt == "py-pytorch" else args.onnx
            w.writerow([rt, model, args.imgsz, args.threads, f"{r['pre']:.3f}", f"{i['mean']:.3f}",
                        f"{i['p50']:.3f}", f"{i['p90']:.3f}", f"{i['p99']:.3f}",
                        f"{r['post']:.3f}", f"{total:.3f}", f"{1000 / total:.2f}"])
            print(f"{rt:16s} inference mean {i['mean']:7.2f} ms  p50 {i['p50']:7.2f}  p90 {i['p90']:7.2f}")


if __name__ == "__main__":
    main()
