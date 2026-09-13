// Detection: one detected object (box + confidence + class). DetectionResult:
// a full detection run's output (detections + image size + timing) — not
// currently used by main.cpp, which builds a plain vector<Detection> directly
// (see PostprocessOutputFP16 in src/main.cpp).

#pragma once

#include <string>
#include <vector>

namespace yolov8 {

struct Detection {
    float x1, y1, x2, y2;    // box corners: (x1,y1) top-left, (x2,y2) bottom-right
    float confidence;        // 0.0-1.0
    int class_id;            // index into the class-names list
    std::string class_name;  // human-readable label for class_id

    Detection() = default;

    Detection(float x1, float y1, float x2, float y2, float conf, int cls_id)
        : x1(x1), y1(y1), x2(x2), y2(y2), confidence(conf), class_id(cls_id) {}
};

struct DetectionResult {
    std::vector<Detection> detections;
    int image_width;
    int image_height;
    float inference_time_ms;
    float postprocess_time_ms;

    DetectionResult() = default;

    DetectionResult(int w, int h)
        : image_width(w), image_height(h), inference_time_ms(0), postprocess_time_ms(0) {}
};

}  // namespace yolov8
