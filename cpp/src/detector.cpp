#include "detector.hpp"

#include <algorithm>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat letterbox_blob(const cv::Mat& bgr, int input_size, LetterboxInfo& info) {
    const float scale = std::min(static_cast<float>(input_size) / bgr.cols,
                                 static_cast<float>(input_size) / bgr.rows);
    const int new_w = static_cast<int>(std::round(bgr.cols * scale));
    const int new_h = static_cast<int>(std::round(bgr.rows * scale));
    info.scale = scale;
    info.pad_x = (input_size - new_w) / 2.0f;
    info.pad_y = (input_size - new_h) / 2.0f;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int top = static_cast<int>(std::round(info.pad_y - 0.1f));
    const int left = static_cast<int>(std::round(info.pad_x - 0.1f));
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, input_size - new_h - top, left,
                       input_size - new_w - left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    info.pad_x = static_cast<float>(left);
    info.pad_y = static_cast<float>(top);

    // HWC BGR uint8 -> NCHW RGB float32 / 255
    return cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(), cv::Scalar(),
                                  /*swapRB=*/true, /*crop=*/false, CV_32F);
}

std::vector<Detection> decode_yolo(const float* data, int num_channels, int num_anchors,
                                   const LetterboxInfo& info, const cv::Size& orig,
                                   float conf_thr, float iou_thr) {
    const int num_classes = num_channels - 4;
    std::vector<cv::Rect> boxes;          // used by NMS (int rect, class-offset)
    std::vector<cv::Rect2f> boxes_f;      // real boxes
    std::vector<float> scores;
    std::vector<int> class_ids;

    // Memory layout is channel-major: data[c * num_anchors + a].
    for (int a = 0; a < num_anchors; ++a) {
        int best_cls = 0;
        float best = data[4 * num_anchors + a];
        for (int c = 1; c < num_classes; ++c) {
            const float s = data[(4 + c) * num_anchors + a];
            if (s > best) { best = s; best_cls = c; }
        }
        if (best < conf_thr) continue;

        const float cx = data[0 * num_anchors + a];
        const float cy = data[1 * num_anchors + a];
        const float w  = data[2 * num_anchors + a];
        const float h  = data[3 * num_anchors + a];

        // Undo letterbox, clip to image.
        float x1 = (cx - w / 2 - info.pad_x) / info.scale;
        float y1 = (cy - h / 2 - info.pad_y) / info.scale;
        float x2 = (cx + w / 2 - info.pad_x) / info.scale;
        float y2 = (cy + h / 2 - info.pad_y) / info.scale;
        x1 = std::clamp(x1, 0.f, static_cast<float>(orig.width));
        y1 = std::clamp(y1, 0.f, static_cast<float>(orig.height));
        x2 = std::clamp(x2, 0.f, static_cast<float>(orig.width));
        y2 = std::clamp(y2, 0.f, static_cast<float>(orig.height));

        cv::Rect2f r(x1, y1, x2 - x1, y2 - y1);
        // Class-aware NMS trick: shift boxes of different classes apart.
        const float off = static_cast<float>(best_cls) * 8192.f;
        boxes.emplace_back(cv::Rect(cv::Point(int(x1 + off), int(y1 + off)),
                                    cv::Point(int(x2 + off), int(y2 + off))));
        boxes_f.push_back(r);
        scores.push_back(best);
        class_ids.push_back(best_cls);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_thr, iou_thr, keep);

    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int i : keep) out.push_back({boxes_f[i], class_ids[i], scores[i]});
    return out;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets,
                     const std::vector<std::string>& names) {
    for (const auto& d : dets) {
        const int k = d.class_id + 1;  // +1 so class 0 is not black
        const cv::Scalar color((k * 67) % 255, (k * 131) % 255, (k * 199) % 255);
        cv::rectangle(img, d.box, color, 2);
        std::string label = (d.class_id < static_cast<int>(names.size())
                                 ? names[d.class_id]
                                 : std::to_string(d.class_id));
        label += " " + cv::format("%.2f", d.score);
        int base = 0;
        const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const cv::Point tl(static_cast<int>(d.box.x),
                           std::max(0, static_cast<int>(d.box.y) - ts.height - 4));
        cv::rectangle(img, cv::Rect(tl, cv::Size(ts.width + 4, ts.height + 4)), color, cv::FILLED);
        cv::putText(img, label, tl + cv::Point(2, ts.height + 1), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 255, 255), 1);
    }
}

std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    return names;
}
