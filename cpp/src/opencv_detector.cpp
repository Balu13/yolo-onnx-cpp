#include "opencv_detector.hpp"

#include <chrono>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

OpenCVDetector::OpenCVDetector(const std::string& model_path, int input_size,
                               float conf_thr, float iou_thr, bool use_cuda)
    : input_size_(input_size), conf_thr_(conf_thr), iou_thr_(iou_thr) {
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (net_.empty()) throw std::runtime_error("Failed to load " + model_path);
    if (use_cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

std::vector<Detection> OpenCVDetector::detect(const cv::Mat& bgr, StageTimes* t) {
    auto t0 = Clock::now();
    LetterboxInfo info{};
    cv::Mat blob = letterbox_blob(bgr, input_size_, info);
    if (t) t->pre_ms = ms_since(t0);

    t0 = Clock::now();
    net_.setInput(blob);
    cv::Mat out = net_.forward();  // [1, 4+nc, anchors]
    if (t) t->infer_ms = ms_since(t0);

    t0 = Clock::now();
    if (!out.isContinuous()) out = out.clone();
    auto dets = decode_yolo(out.ptr<float>(), out.size[1], out.size[2], info, bgr.size(),
                            conf_thr_, iou_thr_);
    if (t) t->post_ms = ms_since(t0);
    return dets;
}
