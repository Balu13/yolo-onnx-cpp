// yolo_infer: run a YOLO ONNX model from C++ and benchmark it.
//
//   yolo_infer --model best.onnx --image img.jpg --backend ort --out result.jpg
//   yolo_infer --model best.onnx --image img.jpg --backend opencv --bench 200 --warmup 20
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "detector.hpp"
#include "opencv_detector.hpp"
#ifdef WITH_ORT
#include "ort_detector.hpp"
#endif

struct Args {
    std::string model, image, backend = "ort", out, names, csv;
    int imgsz = 640, bench = 0, warmup = 10, threads = 0;
    float conf = 0.25f, iou = 0.45f;
    bool cuda = false;
};

static void usage() {
    std::cerr << "Usage: yolo_infer --model M.onnx --image IMG [--backend ort|opencv]\n"
                 "  [--out out.jpg] [--names classes.txt] [--imgsz 640] [--conf 0.25] [--iou 0.45]\n"
                 "  [--bench N] [--warmup W] [--threads T] [--cuda] [--csv results.csv]\n";
}

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(1); }
            return argv[++i];
        };
        if (k == "--model") a.model = next();
        else if (k == "--image") a.image = next();
        else if (k == "--backend") a.backend = next();
        else if (k == "--out") a.out = next();
        else if (k == "--names") a.names = next();
        else if (k == "--csv") a.csv = next();
        else if (k == "--imgsz") a.imgsz = std::stoi(next());
        else if (k == "--bench") a.bench = std::stoi(next());
        else if (k == "--warmup") a.warmup = std::stoi(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--iou") a.iou = std::stof(next());
        else if (k == "--cuda") a.cuda = true;
        else { usage(); std::exit(1); }
    }
    if (a.model.empty() || a.image.empty()) { usage(); std::exit(1); }
    return a;
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(idx), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}
static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);

    std::unique_ptr<Detector> det;
    try {
        if (a.backend == "opencv") {
            det = std::make_unique<OpenCVDetector>(a.model, a.imgsz, a.conf, a.iou, a.cuda);
        } else if (a.backend == "ort") {
#ifdef WITH_ORT
            det = std::make_unique<OrtDetector>(a.model, a.imgsz, a.conf, a.iou, a.threads, a.cuda);
#else
            std::cerr << "Built without ONNX Runtime (configure with -DONNXRUNTIME_DIR=...)\n";
            return 1;
#endif
        } else {
            usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Init error: " << e.what() << "\n";
        return 1;
    }

    cv::Mat img = cv::imread(a.image);
    if (img.empty()) { std::cerr << "Cannot read " << a.image << "\n"; return 1; }

    // Single run: print detections and optionally save visualisation.
    StageTimes t;
    auto dets = det->detect(img, &t);
    const auto names = a.names.empty() ? std::vector<std::string>{} : load_class_names(a.names);
    std::cout << "[" << det->name() << "] " << dets.size() << " detections\n";
    for (const auto& d : dets) {
        const std::string n = d.class_id < (int)names.size() ? names[d.class_id] : std::to_string(d.class_id);
        std::cout << cv::format("  %-14s %.3f  [%.0f, %.0f, %.0f, %.0f]\n", n.c_str(), d.score,
                                d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height);
    }
    if (!a.out.empty()) {
        cv::Mat vis = img.clone();
        draw_detections(vis, dets, names);
        cv::imwrite(a.out, vis);
        std::cout << "Saved " << a.out << "\n";
    }

    if (a.bench <= 0) return 0;

    // Benchmark: warm-up (lazy init, caches), then timed iterations.
    for (int i = 0; i < a.warmup; ++i) det->detect(img);
    std::vector<double> pre, inf, post, total;
    for (int i = 0; i < a.bench; ++i) {
        StageTimes s;
        det->detect(img, &s);
        pre.push_back(s.pre_ms); inf.push_back(s.infer_ms);
        post.push_back(s.post_ms); total.push_back(s.total());
    }
    std::cout << cv::format(
        "\nBenchmark %s (%d iters, %d warm-up, imgsz %d)\n"
        "  preprocess  mean %7.2f ms\n"
        "  inference   mean %7.2f ms | p50 %7.2f | p90 %7.2f | p99 %7.2f\n"
        "  postprocess mean %7.2f ms\n"
        "  end-to-end  mean %7.2f ms -> %.1f FPS\n",
        det->name().c_str(), a.bench, a.warmup, a.imgsz, mean(pre), mean(inf),
        percentile(inf, 50), percentile(inf, 90), percentile(inf, 99), mean(post), mean(total),
        1000.0 / mean(total));

    if (!a.csv.empty()) {
        const bool exists = std::ifstream(a.csv).good();
        std::ofstream f(a.csv, std::ios::app);
        if (!exists) f << "runtime,model,imgsz,threads,pre_ms,infer_mean_ms,infer_p50_ms,infer_p90_ms,infer_p99_ms,post_ms,total_ms,fps\n";
        f << cv::format("cpp-%s,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f\n",
                        det->name().c_str(), a.model.c_str(), a.imgsz, a.threads, mean(pre),
                        mean(inf), percentile(inf, 50), percentile(inf, 90), percentile(inf, 99),
                        mean(post), mean(total), 1000.0 / mean(total));
    }
    return 0;
}
