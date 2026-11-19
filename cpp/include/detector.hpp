#pragma once
// Common types + pre/post-processing shared by all inference backends.
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2f box;   // in original image pixels
    int class_id;
    float score;
};

// Letterbox metadata so boxes can be mapped back to the original image.
struct LetterboxInfo {
    float scale;
    float pad_x;
    float pad_y;
};

// Timings of one call, milliseconds.
struct StageTimes {
    double pre_ms = 0, infer_ms = 0, post_ms = 0;
    double total() const { return pre_ms + infer_ms + post_ms; }
};

// Resize with unchanged aspect ratio + grey (114) padding, like Ultralytics.
// Returns NCHW float32 blob in [0,1], RGB order.
cv::Mat letterbox_blob(const cv::Mat& bgr, int input_size, LetterboxInfo& info);

// Decode YOLOv8/YOLO11 head output: shape [1, 4 + num_classes, num_anchors]
// (cx, cy, w, h, class scores...). No objectness term.
std::vector<Detection> decode_yolo(const float* data, int num_channels, int num_anchors,
                                   const LetterboxInfo& info, const cv::Size& orig,
                                   float conf_thr, float iou_thr);

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets,
                     const std::vector<std::string>& class_names);

std::vector<std::string> load_class_names(const std::string& path);

// Abstract backend: ONNX Runtime or OpenCV DNN.
class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& bgr, StageTimes* t = nullptr) = 0;
    virtual std::string name() const = 0;
};
