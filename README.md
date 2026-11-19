# YOLO → ONNX → C++: train in Python, deploy and benchmark in C++

End-to-end object-detection pipeline: fine-tune **YOLO11n** in Python (Ultralytics / PyTorch),
export to **ONNX**, and run it from a standalone **C++17** application with two interchangeable
backends: **ONNX Runtime** and **OpenCV DNN**. Includes accuracy-parity checks and latency benchmarks
(mean / p50 / p90 / p99, per-stage timing) against the Python baselines.

![demo](results/demo_ort.jpg)

## Results

<!-- Replace with your numbers: python python/make_report.py prints this table -->
Hardware: `<CPU model>`, `<RAM>`, `<OS>`; YOLO11n @ 640×640, batch 1, FP32, 200 iterations after 20 warm-up.

| runtime | threads | infer mean, ms | p50 | p90 | end-to-end, ms | FPS |
|---|---|---|---|---|---|---|
| cpp-onnxruntime | 4 | … | … | … | … | … |
| cpp-opencv-dnn | – | … | … | … | … | … |
| py-onnxruntime | – | … | … | … | … | … |
| py-pytorch | – | … | … | … | … | … |

![latency](results/latency.png)

**Accuracy** (validation set, `python/check_parity.py`): PyTorch mAP50-95 = `…`, ONNX mAP50-95 = `…`
(|Δ| < 0.01, so the export is lossless).

**Takeaways** (fill in from your measurements), e.g.:
- C++ ONNX Runtime vs PyTorch: `x.x×` faster inference on CPU.
- Thread scaling: 1 → 4 threads gives `x.x×`; beyond physical cores there is no gain.
- Pre/post-processing in C++ costs `~x ms`, i.e. `y %` of end-to-end latency.

## Pipeline

```
 dataset ──► train.py (PyTorch) ──► best.pt ──► export.py ──► best.onnx + classes.txt
                                       │                            │
                          check_parity.py (mAP PyTorch vs ONNX)     ▼
                                                     C++ yolo_infer
                                       letterbox ─► ORT / OpenCV DNN ─► decode + NMS
                                                            │
                                         benchmark.csv ─► make_report.py ─► table + chart
```

## Repository layout

```
python/   train.py, export.py, check_parity.py, benchmark.py, make_report.py
cpp/      CMakeLists.txt, include/, src/   (yolo_infer executable)
notebooks/train_colab.ipynb                 (training on a free Colab GPU)
scripts/run_bench.sh                        (all benchmarks -> results/)
results/  benchmark.csv, benchmark.md, latency.png, demo images
```

## Quick start

### 1. Python: train and export

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python python/train.py --data african-wildlife.yaml --epochs 50        # add --device 0 on GPU
python python/export.py --weights models/best.pt   # -> models/best.onnx
python python/check_parity.py --pt models/best.pt \
       --onnx models/best.onnx --data african-wildlife.yaml
```
No GPU? Use `notebooks/train_colab.ipynb` and download `best.pt`, `best.onnx`, `classes.txt`.

### 2. C++: build

Dependencies: CMake ≥ 3.16, a C++17 compiler, OpenCV (tested 4.12; for the **OpenCV DNN backend**
4.6 — the Ubuntu 24.04 apt version — fails to parse YOLO11's attention block, while the ONNX Runtime
backend works with it), and the ONNX Runtime prebuilt package from
[github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
(`onnxruntime-linux-x64-<ver>.tgz` or `onnxruntime-win-x64-<ver>.zip`).

Linux / WSL:
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=$HOME/onnxruntime-linux-x64-<ver>
cmake --build cpp/build -j
```

Windows (Visual Studio 2022, x64 Native Tools prompt):
```bat
cmake -S cpp -B cpp\build -G "Visual Studio 17 2022" -A x64 ^
      -DOpenCV_DIR=C:\opencv\build ^
      -DONNXRUNTIME_DIR=C:\onnxruntime-win-x64-<ver>
cmake --build cpp\build --config Release
```
(add `C:\opencv\build\x64\vc16\bin` to `PATH` so the OpenCV DLL is found)

### 3. Run and benchmark

```bash
./cpp/build/yolo_infer --model models/best.onnx --image assets/test.jpg \
    --names models/classes.txt --backend ort --out results/demo_ort.jpg
./cpp/build/yolo_infer --model models/best.onnx --image assets/test.jpg --backend ort \
    --threads 4 --bench 200 --warmup 20
scripts/run_bench.sh models/best.pt models/best.onnx assets/test.jpg
```

## Implementation notes

- **Pre-processing** reproduces Ultralytics exactly: aspect-preserving resize + grey (114) letterbox,
  BGR→RGB, /255, HWC→NCHW. Mismatched pre-processing is the most common cause of accuracy loss
  after export, which `check_parity.py` guards against.
- **Output decoding**: YOLOv8/11 heads emit `[1, 4 + C, 8400]` (cx, cy, w, h + class scores, no
  objectness). The decoder reads the channel-major tensor in place (no transpose copy), maps boxes
  back through the letterbox and runs class-aware NMS (`cv::dnn::NMSBoxes` with per-class offsets).
- **Zero-copy input**: the ORT input tensor wraps the OpenCV blob memory directly.
- **Benchmark methodology**: Release build, warm-up runs excluded, per-stage timing with
  `std::chrono::steady_clock`, percentiles reported because tail latency matters for real-time use.
- **Backend abstraction**: `Detector` interface → `OrtDetector` / `OpenCVDetector`; adding TensorRT
  or OpenVINO is one new class.

## Possible extensions

- INT8 static quantisation (`onnxruntime.quantization`) and the speed/mAP trade-off.
- GPU: CUDA / TensorRT execution provider (`--cuda`), FP16 export (`export.py --half`).
- Video / webcam stream with a producer–consumer thread pipeline.
- Unit tests (GoogleTest) for letterbox and decoding; GitHub Actions CI build.
