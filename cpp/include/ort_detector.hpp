#pragma once
#include <onnxruntime_cxx_api.h>

#include <memory>

#include "detector.hpp"

class OrtDetector : public Detector {
public:
    OrtDetector(const std::string& model_path, int input_size, float conf_thr,
                float iou_thr, int intra_threads = 0, bool use_cuda = false);
    std::vector<Detection> detect(const cv::Mat& bgr, StageTimes* t = nullptr) override;
    std::string name() const override { return "onnxruntime"; }

private:
    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo mem_info_;
    std::string input_name_, output_name_;
    int input_size_;
    float conf_thr_, iou_thr_;
};
