#!/usr/bin/env bash
# Runs every benchmark into results/benchmark.csv and builds the report.
# Usage: scripts/run_bench.sh <best.pt> <best.onnx> <image.jpg>
set -euo pipefail

PT=${1:?best.pt}
ONNX=${2:?best.onnx}
IMG=${3:?image}
BIN=${BIN:-cpp/build/yolo_infer}   # Windows/VS: BIN=cpp/build/Release/yolo_infer.exe
CSV=results/benchmark.csv
PY=${PYTHON:-python3}
ITERS=${ITERS:-200}
WARMUP=${WARMUP:-20}

mkdir -p results
rm -f "$CSV"

"$PY" python/benchmark.py --pt "$PT" --onnx "$ONNX" --image "$IMG" \
    --iters "$ITERS" --warmup "$WARMUP" --csv "$CSV"

"$BIN" --model "$ONNX" --image "$IMG" --names models/classes.txt --backend opencv \
    --bench "$ITERS" --warmup "$WARMUP" --csv "$CSV" --out results/demo_opencv.jpg

if "$BIN" --model "$ONNX" --image "$IMG" --backend ort >/dev/null 2>&1; then
    for T in 1 2 4 0; do   # 0 = ORT default (all physical cores)
        "$BIN" --model "$ONNX" --image "$IMG" --names models/classes.txt --backend ort \
            --threads "$T" --bench "$ITERS" --warmup "$WARMUP" --csv "$CSV" \
            --out results/demo_ort.jpg
    done
else
    echo "ONNX Runtime backend not built, skipping."
fi

"$PY" python/make_report.py --csv "$CSV"
