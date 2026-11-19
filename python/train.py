"""Fine-tune a YOLO detector on a dataset (Ultralytics).

Examples:
    python python/train.py --data african-wildlife.yaml --epochs 50
    python python/train.py --data path/to/my_dataset/data.yaml --model yolo11s.pt --device 0
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to start from")
    p.add_argument("--data", default="african-wildlife.yaml", help="dataset yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'0' for GPU, 'cpu' for CPU, default auto")
    p.add_argument("--name", default="yolo11n_wildlife")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
        patience=15,  # early stopping
        seed=0,
        deterministic=True,
    )
    metrics = model.val()  # validation on the best weights
    print(f"mAP50-95: {metrics.box.map:.4f}  mAP50: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
