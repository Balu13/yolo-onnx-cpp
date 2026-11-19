"""Export trained weights to ONNX, sanity-check the graph, dump class names.

    python python/export.py --weights runs/yolo11n_wildlife/weights/best.pt
Creates: models/best.onnx, models/classes.txt
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--half", action="store_true", help="FP16 export (GPU inference only)")
    p.add_argument("--out-dir", default="models")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    onnx_path = Path(
        model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                     simplify=True, dynamic=False, half=args.half)
    )
    dst = out_dir / onnx_path.name
    shutil.copy(onnx_path, dst)

    # Class names, one per line, in id order -> used by the C++ app.
    names = model.names
    (out_dir / "classes.txt").write_text("\n".join(names[i] for i in range(len(names))) + "\n")

    # 1) structural check, 2) run once with ONNX Runtime
    onnx.checker.check_model(onnx.load(str(dst)))
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: np.zeros((1, 3, args.imgsz, args.imgsz), np.float32)})[0]
    print(f"OK  {dst}  input {inp.name}{inp.shape} -> output {out.shape}")
    print(f"    expected output [1, 4 + {len(names)} classes, anchors]")


if __name__ == "__main__":
    main()
