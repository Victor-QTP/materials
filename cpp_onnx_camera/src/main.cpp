// ONNX Runtime-based detector (CUDA, FP16 I/O): batch file/directory
// mode, or live mode from a local camera or an RTSP/video stream. Model input
// size is fixed at 640x640 and must match the model's export resolution.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "../include/yolov8/detection.hpp"
#include "../include/yolov8/utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <map>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

using namespace yolov8;

// ============================================================================
// Visualization (Tableau10 categorical palette)
// ============================================================================

std::map<int, cv::Scalar> class_colors;

void InitColors() {
    class_colors[0] = cv::Scalar(180, 119, 31);   // pedestrian    - blue   #1f77b4
    class_colors[1] = cv::Scalar(14, 127, 255);   // rider         - orange #ff7f0e
    class_colors[2] = cv::Scalar(44, 160, 44);    // car           - green  #2ca02c
    class_colors[3] = cv::Scalar(40, 39, 214);    // truck         - red    #d62728
    class_colors[4] = cv::Scalar(189, 103, 148);  // bus           - purple #9467bd
    class_colors[5] = cv::Scalar(75, 86, 140);    // train         - brown  #8c564b
    class_colors[6] = cv::Scalar(194, 119, 227);  // motorcycle    - pink   #e377c2
    class_colors[7] = cv::Scalar(127, 127, 127);  // bicycle       - gray   #7f7f7f
    class_colors[8] = cv::Scalar(34, 189, 188);   // traffic light - olive  #bcbd22
    class_colors[9] = cv::Scalar(207, 190, 23);   // traffic sign  - cyan   #17becf
}

cv::Scalar GetContrastingTextColor(const cv::Scalar& bg) {
    double luminance = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2];  // bg is B,G,R
    return luminance > 140.0 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
}

void DrawDetections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::Scalar color = class_colors.count(det.class_id) ? class_colors[det.class_id] : cv::Scalar(150, 150, 150);
        cv::Scalar text_color = GetContrastingTextColor(color);

        cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), color, 2, cv::LINE_AA);

        std::ostringstream oss;
        oss << det.class_name << " " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label = oss.str();

        int baseline = 0;
        double font_scale = 0.9;
        int thickness = 2;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

        int text_x = det.x1;
        int text_y = det.y1 - 4;
        if (text_y - text_size.height < 0) text_y = det.y1 + text_size.height + 4;

        cv::rectangle(img, cv::Point(text_x, text_y - text_size.height - 4),
                      cv::Point(text_x + text_size.width + 4, text_y + baseline),
                      color, -1, cv::LINE_AA);

        cv::putText(img, label, cv::Point(text_x + 2, text_y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv::LINE_AA);
    }
}

std::string FormatFixed(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void DrawMetricsOverlay(cv::Mat& img, float inference_time_ms, int num_detections) {
    float fps = inference_time_ms > 0.0f ? 1000.0f / inference_time_ms : 0.0f;

    std::vector<std::string> metrics = {
        "FPS: " + FormatFixed(fps, 1),
        "Inference: " + FormatFixed(inference_time_ms, 1) + "ms",
        "Detections: " + std::to_string(num_detections),
        "Runtime: ONNX Runtime (CUDA GPU, FP16 I/O)",
        "Input: 640x640",
    };

    const int x_start = 10;
    const int y_start = 30;
    const int line_height = 35;
    const double font_scale = 0.7;
    const int thickness = 2;
    const double bg_alpha = 0.7;

    for (size_t i = 0; i < metrics.size(); ++i) {
        int y_pos = y_start + static_cast<int>(i) * line_height;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(metrics[i], cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

        cv::Rect bg(x_start - 5, y_pos - text_size.height - 5,
                    text_size.width + 10, text_size.height + baseline + 10);

        cv::Mat overlay = img.clone();
        cv::rectangle(overlay, bg, cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, img);

        cv::putText(img, metrics[i], cv::Point(x_start, y_pos),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    }
}

// ============================================================================
// Filesystem helpers
// ============================================================================

bool PathIsDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

bool PathExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

void MakeDirsRecursive(const std::string& path) {
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            if (!current.empty() && !PathExists(current)) {
                mkdir(current.c_str(), 0755);
            }
        }
    }
}

std::string StripTrailingSlash(const std::string& path) {
    if (path.size() > 1 && path.back() == '/') return path.substr(0, path.size() - 1);
    return path;
}

std::string ParentDir(const std::string& path) {
    size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? "." : path.substr(0, pos);
}

std::string BaseName(const std::string& path) {
    size_t pos = path.find_last_of('/');
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

std::string StripExtension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    return (pos == std::string::npos) ? filename : filename.substr(0, pos);
}

bool HasImageExtension(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const char* ext : {".jpg", ".jpeg", ".png", ".bmp"}) {
        size_t elen = std::strlen(ext);
        if (lower.size() >= elen && lower.compare(lower.size() - elen, elen, ext) == 0) return true;
    }
    return false;
}

std::vector<std::string> ListImages(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) return files;

    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (HasImageExtension(name)) files.push_back(dir + "/" + name);
    }
    closedir(d);

    std::sort(files.begin(), files.end());
    return files;
}

std::string JsonEscape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

// ============================================================================
// Output writers
// ============================================================================

void WriteYoloLabels(const std::string& labels_dir, const std::string& image_filename,
                      const std::vector<Detection>& detections, int img_width, int img_height) {
    std::string out_path = labels_dir + "/" + StripExtension(image_filename) + ".txt";
    std::ofstream f(out_path);

    for (const auto& det : detections) {
        float cx = ((det.x1 + det.x2) / 2.0f) / img_width;
        float cy = ((det.y1 + det.y2) / 2.0f) / img_height;
        float w = (det.x2 - det.x1) / img_width;
        float h = (det.y2 - det.y1) / img_height;

        f << det.class_id << " "
          << std::fixed << std::setprecision(6)
          << cx << " " << cy << " " << w << " " << h << " "
          << det.confidence << "\n";
    }
}

// Accumulates COCO-format prediction entries; written once at the end.
class CocoWriter {
public:
    void AddImageDetections(int image_id, const std::string& file_name,
                             const std::vector<Detection>& detections) {
        for (const auto& det : detections) {
            std::ostringstream entry;
            entry << "  {\n"
                  << "    \"image_id\": " << image_id << ",\n"
                  << "    \"file_name\": \"" << JsonEscape(file_name) << "\",\n"
                  << "    \"category_id\": " << (det.class_id + 1) << ",\n"
                  << "    \"category_name\": \"" << JsonEscape(det.class_name) << "\",\n"
                  << "    \"bbox\": [" << std::fixed << std::setprecision(2)
                  << det.x1 << ", " << det.y1 << ", "
                  << (det.x2 - det.x1) << ", " << (det.y2 - det.y1) << "],\n"
                  << "    \"score\": " << std::setprecision(6) << det.confidence << "\n"
                  << "  }";
            entries_.push_back(entry.str());
        }
    }

    void WriteToFile(const std::string& path) const {
        std::ofstream f(path);
        f << "[\n";
        for (size_t i = 0; i < entries_.size(); ++i) {
            f << entries_[i];
            if (i + 1 < entries_.size()) f << ",";
            f << "\n";
        }
        f << "]\n";
    }

private:
    std::vector<std::string> entries_;
};

// ============================================================================
// FP16 preprocessing / postprocessing. Model has FLOAT16 I/O tensors
// (elem_type=10), so tensors are built and read as Ort::Float16_t directly.
// ============================================================================

std::vector<Ort::Float16_t> PreprocessImageFP16(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));  // stretch resize, aspect ratio not preserved

    cv::Mat rgb;
    if (img.channels() == 3) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else if (img.channels() == 4) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
    } else if (img.channels() == 1) {
        cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
    } else {
        rgb = resized;
    }

    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    std::vector<float> planar(640 * 640 * 3);
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(640, 640, CV_32F, planar.data() + c * 640 * 640);
    }
    cv::split(float_img, channels);

    std::vector<Ort::Float16_t> output;
    output.reserve(planar.size());
    for (float v : planar) {
        output.emplace_back(v);
    }
    return output;
}

std::vector<Detection> PostprocessOutputFP16(
    const Ort::Float16_t* output, int num_classes,
    const std::vector<std::string>& class_names,
    int img_width, int img_height,
    float confidence_threshold, float nms_threshold, int max_detections) {
    std::vector<Detection> detections;
    const int num_predictions = 8400;  // 80^2 + 40^2 + 20^2 at 640 input
    const int stride = num_predictions;

    for (int i = 0; i < num_predictions; ++i) {
        float cx = static_cast<float>(output[0 * stride + i]);
        float cy = static_cast<float>(output[1 * stride + i]);
        float w  = static_cast<float>(output[2 * stride + i]);
        float h  = static_cast<float>(output[3 * stride + i]);

        float max_conf = -1.0f;
        int class_id = 0;
        for (int c = 0; c < num_classes; ++c) {
            float conf = static_cast<float>(output[(4 + c) * stride + i]);
            if (conf > max_conf) {
                max_conf = conf;
                class_id = c;
            }
        }

        if (max_conf < confidence_threshold) continue;

        float scale_x = static_cast<float>(img_width) / 640.0f;
        float scale_y = static_cast<float>(img_height) / 640.0f;

        float x1 = (cx - w / 2.0f) * scale_x;
        float y1 = (cy - h / 2.0f) * scale_y;
        float x2 = (cx + w / 2.0f) * scale_x;
        float y2 = (cy + h / 2.0f) * scale_y;

        x1 = std::max(0.0f, std::min(static_cast<float>(img_width), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(img_height), y1));
        x2 = std::max(0.0f, std::min(static_cast<float>(img_width), x2));
        y2 = std::max(0.0f, std::min(static_cast<float>(img_height), y2));

        if (x2 <= x1 || y2 <= y1) continue;

        Detection det;
        det.x1 = x1; det.y1 = y1; det.x2 = x2; det.y2 = y2;
        det.confidence = max_conf;
        det.class_id = class_id;
        det.class_name = (class_id >= 0 && class_id < static_cast<int>(class_names.size()))
                         ? class_names[class_id] : "unknown";
        detections.push_back(det);
    }

    std::vector<size_t> keep = utils::NonMaxSuppression(detections, nms_threshold);
    std::vector<Detection> result;
    for (size_t idx : keep) {
        if (static_cast<int>(result.size()) >= max_detections) break;
        result.push_back(detections[idx]);
    }
    return result;
}

std::vector<std::string> LoadClassNames(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Class names file not found: " + path);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) throw std::runtime_error("No class names loaded from: " + path);
    return names;
}

// ============================================================================
// CLI argument parsing
// ============================================================================

struct Args {
    std::string model;
    std::string classes;
    std::string input;
    float conf = 0.25f;
    float nms = 0.5f;
    int max_det = 300;

    std::string vis_dir;
    std::string labels_dir;
    std::string coco_out;

    bool save_vis = true;
    bool save_yolo = true;
    bool save_coco = true;
    bool show_metrics = true;

    int camera_index = -1;      // -1 = not set; a real device index is always >= 0
    std::string stream_url;     // empty = not set
    bool headless = false;      // skip cv::imshow window; useful for headless edge deployment
    std::string save_video;     // optional path to record annotated output; empty = don't record
};

bool IsLiveMode(const Args& args) {
    return args.camera_index >= 0 || !args.stream_url.empty();
}

void PrintUsage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " --model <path> --classes <path> (--input <path> | --camera <index> | --stream <url>) [options]\n\n"
        "Required:\n"
        "  --model <path>        Path to FLOAT16-I/O ONNX model (640x640 input)\n"
        "  --classes <path>      Path to class names file (one per line)\n\n"
        "Exactly one input source is required:\n"
        "  --input <path>        Image file OR directory of images (batch mode)\n"
        "  --camera <index>      Local camera device index (0 = /dev/video0, live mode)\n"
        "  --stream <url>        RTSP/network video URL (live mode)\n\n"
        "Detection options:\n"
        "  --conf <float>        Confidence threshold (default 0.25)\n"
        "  --nms <float>         NMS IoU threshold (default 0.5)\n"
        "  --max-det <int>       Max detections per image/frame (default 300)\n\n"
        "Batch-mode output options (--input only):\n"
        "  --vis-dir <path>      Annotated image output dir (default: <input>_vis)\n"
        "  --labels-dir <path>   YOLO-format label output dir (default: <input>_yolo_labels)\n"
        "  --coco-out <path>     COCO-format predictions JSON path (default: <input>_coco.json)\n"
        "  --vis / --no-vis      Toggle annotated image output (default: on)\n"
        "  --yolo / --no-yolo    Toggle YOLO-format label output (default: on)\n"
        "  --coco / --no-coco    Toggle COCO-format JSON output (default: on)\n\n"
        "Live-mode options (--camera / --stream only):\n"
        "  --headless            Don't open a display window; console output only\n"
        "  --save-video <path>   Record annotated output to a video file\n\n"
        "Shared options:\n"
        "  --metrics/--no-metrics  Toggle per-frame FPS/timing overlay (default: on; needs --vis)\n"
        "  -h, --help            Show this help\n";
}

bool ParseArgs(int argc, char** argv, Args& args) {
    std::vector<std::string> a(argv + 1, argv + argc);

    auto need_value = [&](size_t& i) -> std::string {
        if (i + 1 >= a.size()) throw std::runtime_error("missing value for " + a[i]);
        return a[++i];
    };

    for (size_t i = 0; i < a.size(); ++i) {
        const std::string& flag = a[i];
        if (flag == "-h" || flag == "--help") return false;
        else if (flag == "--model") args.model = need_value(i);
        else if (flag == "--classes") args.classes = need_value(i);
        else if (flag == "--input") args.input = need_value(i);
        else if (flag == "--camera") args.camera_index = std::stoi(need_value(i));
        else if (flag == "--stream") args.stream_url = need_value(i);
        else if (flag == "--conf") args.conf = std::stof(need_value(i));
        else if (flag == "--nms") args.nms = std::stof(need_value(i));
        else if (flag == "--max-det") args.max_det = std::stoi(need_value(i));
        else if (flag == "--vis-dir") args.vis_dir = need_value(i);
        else if (flag == "--labels-dir") args.labels_dir = need_value(i);
        else if (flag == "--coco-out") args.coco_out = need_value(i);
        else if (flag == "--vis") args.save_vis = true;
        else if (flag == "--no-vis") args.save_vis = false;
        else if (flag == "--yolo") args.save_yolo = true;
        else if (flag == "--no-yolo") args.save_yolo = false;
        else if (flag == "--coco") args.save_coco = true;
        else if (flag == "--no-coco") args.save_coco = false;
        else if (flag == "--metrics") args.show_metrics = true;
        else if (flag == "--no-metrics") args.show_metrics = false;
        else if (flag == "--headless") args.headless = true;
        else if (flag == "--save-video") args.save_video = need_value(i);
        else throw std::runtime_error("unknown argument: " + flag);
    }

    if (args.model.empty() || args.classes.empty()) {
        throw std::runtime_error("--model and --classes are required");
    }

    int sources_given = (!args.input.empty() ? 1 : 0) +
                         (args.camera_index >= 0 ? 1 : 0) +
                         (!args.stream_url.empty() ? 1 : 0);
    if (sources_given != 1) {
        throw std::runtime_error("exactly one of --input, --camera, or --stream is required");
    }

    if (!args.input.empty()) {
        std::string base = PathIsDirectory(args.input)
            ? StripTrailingSlash(args.input)
            : ParentDir(args.input) + "/" + StripExtension(BaseName(args.input));

        if (args.vis_dir.empty()) args.vis_dir = base + "_vis";
        if (args.labels_dir.empty()) args.labels_dir = base + "_yolo_labels";
        if (args.coco_out.empty()) args.coco_out = base + "_coco.json";
    }

    return true;
}

// ============================================================================
// Live mode: camera or network stream, read continuously via cv::VideoCapture
// ============================================================================

void RunLiveMode(Ort::Session& session, const Ort::MemoryInfo& memory_info,
                  const char* input_names[], const char* output_names[],
                  const std::vector<std::string>& class_names, int num_classes,
                  const Args& args) {
    cv::VideoCapture cap;
    std::string source_desc;
    if (args.camera_index >= 0) {
        cap.open(args.camera_index);  // local device via V4L2, e.g. index 0 -> /dev/video0
        source_desc = "camera index " + std::to_string(args.camera_index);
    } else {
        cap.open(args.stream_url, cv::CAP_FFMPEG);  // RTSP URL or local video file path
        source_desc = args.stream_url;
    }
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to open video source: " + source_desc);
    }
    std::cout << "Opened video source: " << source_desc << std::endl;

    // First CUDA session.Run() call pays a one-time setup cost; warm up on
    // one real frame before the timed loop so it doesn't skew reported FPS.
    {
        cv::Mat warmup_frame;
        if (cap.read(warmup_frame) && !warmup_frame.empty()) {
            std::vector<Ort::Float16_t> warmup_input = PreprocessImageFP16(warmup_frame);
            std::array<int64_t, 4> warmup_shape{1, 3, 640, 640};
            Ort::Value warmup_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                memory_info, warmup_input.data(), warmup_input.size(),
                warmup_shape.data(), warmup_shape.size());
            session.Run(Ort::RunOptions{nullptr}, input_names, &warmup_tensor, 1, output_names, 1);
            std::cout << "Warmup frame processed.\n";
        }
    }

    cv::VideoWriter writer;
    bool recording = !args.save_video.empty();  // VideoWriter opens lazily once frame size is known

    if (!args.headless) {
        cv::namedWindow("onnx_camera - live", cv::WINDOW_NORMAL);
    }

    long frame_count = 0;
    double total_inference_ms = 0.0;
    auto session_start = std::chrono::high_resolution_clock::now();

    std::cout << "Press 'q' or ESC in the window to stop"
              << (args.headless ? " (headless: Ctrl+C to stop)." : ".") << "\n\n";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "Video source ended or frame read failed — stopping.\n";
            break;
        }
        int img_width = frame.cols;
        int img_height = frame.rows;

        std::vector<Ort::Float16_t> input_data = PreprocessImageFP16(frame);
        std::array<int64_t, 4> input_shape{1, 3, 640, 640};
        Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        auto start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        auto end = std::chrono::high_resolution_clock::now();
        float inference_ms = std::chrono::duration<float, std::milli>(end - start).count();

        const Ort::Float16_t* output_data = output_tensors[0].GetTensorData<Ort::Float16_t>();

        auto detections = PostprocessOutputFP16(
            output_data, num_classes, class_names, img_width, img_height,
            args.conf, args.nms, args.max_det);

        ++frame_count;
        total_inference_ms += inference_ms;

        DrawDetections(frame, detections);
        if (args.show_metrics) {
            DrawMetricsOverlay(frame, inference_ms, static_cast<int>(detections.size()));
        }

        if (recording) {
            if (!writer.isOpened()) {
                double capture_fps = cap.get(cv::CAP_PROP_FPS);
                if (capture_fps <= 1.0 || capture_fps > 240.0) capture_fps = 25.0;  // unreliable FPS fallback
                writer.open(args.save_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            capture_fps, cv::Size(img_width, img_height));
                if (!writer.isOpened()) {
                    std::cerr << "Warning: failed to open --save-video output ("
                              << args.save_video << "), continuing without recording.\n";
                    recording = false;
                } else {
                    std::cout << "Recording annotated output to: " << args.save_video << "\n";
                }
            }
            if (recording) writer.write(frame);
        }

        if (!args.headless) {
            cv::imshow("onnx_camera - live", frame);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {  // 27 = ESC
                std::cout << "Quit key pressed.\n";
                break;
            }
        }

        if (frame_count % 30 == 0) {
            double fps = inference_ms > 0.0 ? 1000.0 / inference_ms : 0.0;
            std::cout << "[frame " << frame_count << "] " << detections.size()
                      << " detections, " << FormatFixed(inference_ms, 1) << " ms ("
                      << FormatFixed(fps, 1) << " FPS)\n";
        }
    }

    auto session_end = std::chrono::high_resolution_clock::now();
    double total_wall_s = std::chrono::duration<double>(session_end - session_start).count();
    double avg_inference_ms = frame_count > 0 ? total_inference_ms / frame_count : 0.0;
    double avg_fps = avg_inference_ms > 0.0 ? 1000.0 / avg_inference_ms : 0.0;

    std::cout << "\n=== Session Summary ===\n"
              << "  Frames processed:       " << frame_count << "\n"
              << "  Average inference time: " << FormatFixed(avg_inference_ms, 1) << " ms\n"
              << "  Average FPS:            " << FormatFixed(avg_fps, 1) << "\n"
              << "  Total wall time:        " << FormatFixed(total_wall_s, 1) << " s\n";

    if (!args.headless) cv::destroyAllWindows();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    Args args;
    try {
        if (!ParseArgs(argc, argv, args)) {
            PrintUsage(argv[0]);
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        auto class_names = LoadClassNames(args.classes);
        int num_classes = static_cast<int>(class_names.size());

        std::cout << "Loading model (FLOAT16 I/O): " << args.model << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_camera");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptionsV2* cuda_options_raw = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options_raw));
        std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>
            cuda_options(cuda_options_raw, Ort::GetApi().ReleaseCUDAProviderOptions);
        session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

        Ort::Session session(env, args.model.c_str(), session_options);
        std::cout << "Model loaded successfully." << std::endl;

        {
            Ort::AllocatorWithDefaultOptions allocator;
            Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
            auto tinfo = type_info.GetTensorTypeAndShapeInfo();
            if (tinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                std::cerr << "Error: model input elem_type=" << tinfo.GetElementType()
                          << ", expected FLOAT16 (10). This tool requires a "
                          << "FLOAT16-I/O model." << std::endl;
                return 1;
            }
        }

        InitColors();

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* input_names[] = {"images"};
        const char* output_names[] = {"output0"};

        if (IsLiveMode(args)) {
            RunLiveMode(session, memory_info, input_names, output_names, class_names, num_classes, args);
            return 0;
        }

        bool batch_mode = PathIsDirectory(args.input);
        std::vector<std::string> image_paths = batch_mode
            ? ListImages(args.input)
            : std::vector<std::string>{args.input};

        if (image_paths.empty()) {
            std::cerr << "No images found at: " << args.input << std::endl;
            return 1;
        }

        std::cout << (batch_mode ? "Batch mode: " : "Single image: ")
                   << image_paths.size() << " image(s)\n";
        if (args.save_vis) std::cout << "  Vis output:    " << args.vis_dir << "\n";
        if (args.save_yolo) std::cout << "  Labels output: " << args.labels_dir << "\n";
        if (args.save_coco) std::cout << "  COCO output:   " << args.coco_out << "\n";
        std::cout << std::endl;

        if (args.save_vis) MakeDirsRecursive(args.vis_dir);
        if (args.save_yolo) MakeDirsRecursive(args.labels_dir);
        if (args.save_coco) MakeDirsRecursive(ParentDir(args.coco_out));

        try {
            std::cout << "Warming up (first CUDA inference call pays a one-time "
                       << "context/kernel-selection cost)...\n";
            cv::Mat warmup_img = cv::imread(image_paths[0]);
            if (!warmup_img.empty()) {
                std::vector<Ort::Float16_t> warmup_input = PreprocessImageFP16(warmup_img);
                std::array<int64_t, 4> warmup_shape{1, 3, 640, 640};
                Ort::Value warmup_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    memory_info, warmup_input.data(), warmup_input.size(),
                    warmup_shape.data(), warmup_shape.size());
                session.Run(Ort::RunOptions{nullptr}, input_names, &warmup_tensor, 1, output_names, 1);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warmup failed: " << e.what() << std::endl;
        }

        CocoWriter coco;
        int total_detections = 0;
        double total_inference_ms = 0.0;
        auto batch_start = std::chrono::high_resolution_clock::now();

        for (size_t idx = 0; idx < image_paths.size(); ++idx) {
            const std::string& image_path = image_paths[idx];
            std::string filename = BaseName(image_path);

            try {
                cv::Mat img = cv::imread(image_path);
                if (img.empty()) throw std::runtime_error("Failed to load image: " + image_path);
                int img_width = img.cols;
                int img_height = img.rows;

                std::vector<Ort::Float16_t> input_data = PreprocessImageFP16(img);
                std::array<int64_t, 4> input_shape{1, 3, 640, 640};
                Ort::Value input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    memory_info, input_data.data(), input_data.size(),
                    input_shape.data(), input_shape.size());

                auto start = std::chrono::high_resolution_clock::now();
                auto output_tensors = session.Run(
                    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
                auto end = std::chrono::high_resolution_clock::now();
                float inference_ms = std::chrono::duration<float, std::milli>(end - start).count();

                const Ort::Float16_t* output_data = output_tensors[0].GetTensorData<Ort::Float16_t>();
                auto detections = PostprocessOutputFP16(
                    output_data, num_classes, class_names, img_width, img_height,
                    args.conf, args.nms, args.max_det);

                total_detections += static_cast<int>(detections.size());
                total_inference_ms += inference_ms;

                std::cout << "[" << (idx + 1) << "/" << image_paths.size() << "] "
                          << filename << ": " << detections.size()
                          << " detections, " << std::fixed << std::setprecision(1)
                          << inference_ms << " ms\n";

                if (args.save_vis) {
                    DrawDetections(img, detections);
                    if (args.show_metrics) {
                        DrawMetricsOverlay(img, inference_ms, static_cast<int>(detections.size()));
                    }
                    cv::imwrite(args.vis_dir + "/" + filename, img);
                }

                if (args.save_yolo) {
                    WriteYoloLabels(args.labels_dir, filename, detections, img_width, img_height);
                }

                if (args.save_coco) {
                    coco.AddImageDetections(static_cast<int>(idx) + 1, filename, detections);
                }
            } catch (const std::exception& e) {
                std::cerr << "  Failed on " << filename << ": " << e.what() << std::endl;
            }
        }

        if (args.save_coco) coco.WriteToFile(args.coco_out);

        auto batch_end = std::chrono::high_resolution_clock::now();
        double total_wall_s = std::chrono::duration<double>(batch_end - batch_start).count();
        double avg_inference_ms = total_inference_ms / image_paths.size();
        double avg_fps = avg_inference_ms > 0.0 ? 1000.0 / avg_inference_ms : 0.0;

        std::cout << "\n=== Performance Metrics (separate warmup call already excluded) ===\n"
                  << "  Average FPS:            " << FormatFixed(avg_fps, 1) << "\n"
                  << "  Average inference time: " << FormatFixed(avg_inference_ms, 1) << " ms\n"
                  << "  Total images:           " << image_paths.size() << "\n"
                  << "  Total wall time:        " << FormatFixed(total_wall_s, 1) << " s (includes I/O, not just inference)\n"
                  << "\n=== Summary ===\n"
                  << "Images processed:  " << image_paths.size() << "\n"
                  << "Total detections:  " << total_detections << "\n";
        if (args.save_vis) std::cout << "Vis images:        " << args.vis_dir << "\n";
        if (args.save_yolo) std::cout << "YOLO labels:       " << args.labels_dir << "\n";
        if (args.save_coco) std::cout << "COCO predictions:  " << args.coco_out << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
