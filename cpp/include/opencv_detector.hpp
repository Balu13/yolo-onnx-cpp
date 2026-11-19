#pragma once
#include <opencv2/dnn.hpp>

#include "detector.hpp"

class OpenCVDetector : public Detector {
public:
    OpenCVDetector(const std::string& model_path, int input_size, float conf_thr,
                   float iou_thr, bool use_cuda = false);
    std::vector<Detection> detect(const cv::Mat& bgr, StageTimes* t = nullptr) override;
    std::string name() const override { return "opencv-dnn"; }

private:
    cv::dnn::Net net_;
    int input_size_;
    float conf_thr_, iou_thr_;
};
