#include "ort_detector.hpp"

#include <array>
#include <chrono>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

#ifdef _WIN32
// ORT on Windows wants a wide-char path.
static std::wstring to_ort_path(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#else
static const std::string& to_ort_path(const std::string& s) { return s; }
#endif

OrtDetector::OrtDetector(const std::string& model_path, int input_size, float conf_thr,
                         float iou_thr, int intra_threads, bool use_cuda)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_size_(input_size), conf_thr_(conf_thr), iou_thr_(iou_thr) {
    opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (intra_threads > 0) opts_.SetIntraOpNumThreads(intra_threads);
    if (use_cuda) {
        OrtCUDAProviderOptions cuda{};
        opts_.AppendExecutionProvider_CUDA(cuda);  // needs onnxruntime-gpu build
    }
    session_ = std::make_unique<Ort::Session>(env_, to_ort_path(model_path).c_str(), opts_);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();
}

std::vector<Detection> OrtDetector::detect(const cv::Mat& bgr, StageTimes* t) {
    auto t0 = Clock::now();
    LetterboxInfo info{};
    cv::Mat blob = letterbox_blob(bgr, input_size_, info);  // NCHW float32, contiguous
    const std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem_info_, blob.ptr<float>(), blob.total(), shape.data(), shape.size());
    if (t) t->pre_ms = ms_since(t0);

    t0 = Clock::now();
    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    if (t) t->infer_ms = ms_since(t0);

    t0 = Clock::now();
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, 4+nc, A]
    const float* data = outputs[0].GetTensorData<float>();
    auto dets = decode_yolo(data, static_cast<int>(out_shape[1]), static_cast<int>(out_shape[2]),
                            info, bgr.size(), conf_thr_, iou_thr_);
    if (t) t->post_ms = ms_since(t0);
    return dets;
}
