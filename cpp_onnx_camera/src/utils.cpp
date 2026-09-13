// Implements the postprocessing helpers declared in utils.hpp: ComputeIoU,
// NonMaxSuppression, FilterByConfidence, SortByConfidence.

#include "../include/yolov8/utils.hpp"
#include "../include/yolov8/detection.hpp"
#include <algorithm>
#include <cmath>

namespace yolov8 {
namespace utils {

float ComputeIoU(const Detection& box1, const Detection& box2) {
    float inter_x1 = std::max(box1.x1, box2.x1);
    float inter_y1 = std::max(box1.y1, box2.y1);
    float inter_x2 = std::min(box1.x2, box2.x2);
    float inter_y2 = std::min(box1.y2, box2.y2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                       std::max(0.0f, inter_y2 - inter_y1);

    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    float union_area = area1 + area2 - inter_area;  // inclusion-exclusion

    return union_area > 0.0f ? inter_area / union_area : 0.0f;  // avoid divide-by-zero
}

std::vector<size_t> NonMaxSuppression(
    const std::vector<Detection>& detections,
    float nms_threshold) {
    if (detections.empty()) {
        return {};
    }

    std::vector<size_t> indices(detections.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(),
        [&detections](size_t i, size_t j) {
            return detections[i].confidence > detections[j].confidence;
        });  // descending by confidence

    std::vector<size_t> keep;
    std::vector<bool> suppressed(indices.size(), false);

    // Greedy NMS: take the highest-confidence remaining box, keep it, and
    // suppress every later box that overlaps it by more than nms_threshold.
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[i]) continue;

        size_t idx_i = indices[i];
        keep.push_back(idx_i);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[j]) continue;

            size_t idx_j = indices[j];
            float iou = ComputeIoU(detections[idx_i], detections[idx_j]);

            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return keep;  // indices into the original detections vector
}

std::vector<Detection> FilterByConfidence(
    const std::vector<Detection>& detections,
    float confidence_threshold) {
    std::vector<Detection> filtered;
    for (const auto& det : detections) {
        if (det.confidence >= confidence_threshold) {
            filtered.push_back(det);
        }
    }
    return filtered;
}

void SortByConfidence(std::vector<Detection>& detections) {
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
}

}  // namespace utils
}  // namespace yolov8
