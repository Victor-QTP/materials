// Detection-postprocessing helpers: ComputeIoU (box overlap),
// NonMaxSuppression (drop duplicate overlapping boxes), FilterByConfidence,
// SortByConfidence. Implementations in src/utils.cpp.

#pragma once

#include "detection.hpp"
#include <vector>

namespace yolov8 {
namespace utils {

// IoU = Intersection Area / Union Area
float ComputeIoU(const Detection& box1, const Detection& box2);

// Returns indices (into detections) of boxes that survive NMS.
std::vector<size_t> NonMaxSuppression(
    const std::vector<Detection>& detections,
    float nms_threshold
);

// Returns a new list containing only detections at or above the threshold.
std::vector<Detection> FilterByConfidence(
    const std::vector<Detection>& detections,
    float confidence_threshold
);

// Reorders detections in place, highest confidence first.
void SortByConfidence(std::vector<Detection>& detections);

}  // namespace utils
}  // namespace yolov8
