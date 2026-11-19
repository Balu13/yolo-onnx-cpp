"""Fine-tune a YOLO detector on a dataset (Ultralytics).

Examples:
    python python/train.py --data african-wildlife.yaml --epochs 50
    python python/train.py --data path/to/my_dataset/data.yaml --model yolo11s.pt --device 0
"""
import argparse
import shutil
from pathlib import Path
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

    # Ultralytics may nest the run dir differently between versions -> copy to a fixed path
    best = Path(model.trainer.best)          # .../weights/best.pt
    run_dir = best.parent.parent             # .../yolo11n_wildlife
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    if (run_dir / "results.png").exists():
        shutil.copy(run_dir / "results.png", "results/training_curves.png")
    print(f"Run dir: {run_dir}\n  -> models/best.pt, results/training_curves.png")
    
if __name__ == "__main__":
    main()
