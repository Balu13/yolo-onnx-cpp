"""Accuracy check: ONNX model must give the same mAP as the PyTorch model.

    python python/check_parity.py --pt runs/.../best.pt --onnx models/best.onnx --data african-wildlife.yaml
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    res = {}
    for name, w in [("pytorch", args.pt), ("onnx", args.onnx)]:
        m = YOLO(w, task="detect").val(data=args.data, imgsz=args.imgsz, batch=1,
                                        device="cpu", plots=False, verbose=False)
        res[name] = (m.box.map50, m.box.map)
    for k, (m50, m) in res.items():
        print(f"{k:8s} mAP50 {m50:.4f}  mAP50-95 {m:.4f}")
    d = abs(res["pytorch"][1] - res["onnx"][1])
    print(f"|delta mAP50-95| = {d:.4f}  ->  {'OK' if d < 0.01 else 'CHECK EXPORT'}")


if __name__ == "__main__":
    main()
