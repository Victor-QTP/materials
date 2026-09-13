# Real-Time Inference with YOLOv8 (C++ and ONNX Runtime)

C++ deployment pipeline for YOLOv8-based object detection using ONNX Runtime with the CUDA Execution Provider and FLOAT16 input/output tensors. Supports three input sources: image files/directories, local camera devices and video sources such as RTSP streams or local video files.

Only 640x640 exports are supported by the current implementation. See [Model Input Resolution](#model-input-resolution).

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Models](#models)
- [Model Input Resolution](#model-input-resolution)
- [Usage](#usage)
  - [CLI Reference](#cli-reference)
  - [Reproducing the Singapore Driving Demo](#reproducing-the-singapore-driving-demo)
  - [Live Camera or Stream](#live-camera-or-stream)
- [How It Works](#how-it-works)
- [Roadmap: INT8 Quantization](#roadmap-int8-quantization)

## Requirements

**Platform**
- Current bundled runtime targets Linux x86-64. The C++ inference pipeline is portable to ARM64 platforms such as NVIDIA Jetson with architecture-compatible runtime dependencies.

**Hardware**
- NVIDIA GPU with CUDA 12.x and cuDNN **8.x**. This is a hard version pin, not a suggestion: the vendored ONNX Runtime build is 1.18.0, which links against cuDNN 8. ONNX Runtime 1.18.1 would require cuDNN 9 instead, so upgrading the ORT build without also upgrading cuDNN (or vice versa) breaks the link at runtime.
- A camera device (e.g. `/dev/video0`) only if you intend to use `--camera` mode. Not needed for `--input` (batch) or `--stream` (file) modes.

**Software** (all provided by the Docker image below, per [Installation](#installation))
- CMake ≥ 3.15, C++17 compiler
- OpenCV (apt package `libopencv-dev=4.2.0+dfsg-5`)
- GStreamer: `gstreamer1.0-plugins-good`, `gstreamer1.0-libav`, `gstreamer1.0-tools`: required for OpenCV to open camera/RTSP streams (`--camera`, `--stream`). Not needed for `--input` batch mode.
- `ffmpeg`: only needed for the frame-extraction step in [Reproducing the Demo](#reproducing-the-singapore-driving-demo), not by the compiled binary itself.
- ONNX Runtime GPU 1.18.0: already vendored in `third_party/onnxruntime-linux-x64-gpu-1.18.0/`. Nothing to fetch.
- ONNX Runtime GPU 1.18.0 → CUDA 12.x + cuDNN 8.x
- ONNX Runtime GPU 1.18.1 → CUDA 12.x + cuDNN 9.x

**Optional: Python environment, model conversion only**
Only needed if you're re-exporting a `.pt` checkpoint yourself (see `scripts/convert_pt_to_fp16.py`). Not needed to build or run the C++ detector against the models already checked into `models/`.
```bash
pip install onnx onnxconverter_common
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Plus a local clone of Ultralytics reachable via `sys.path`: this project uses `../ultralytics_repo` directly rather than the pip package, to avoid pulling a second, heavier dependency chain on top of PyTorch (see the conversion scripts' docstrings for details).

## Installation

**1. Build the Docker image.** This project shares one image with its sibling projects; the `Dockerfile` lives in the parent `projects/` directory:
```bash
cd /path/to/projects
docker build -f Dockerfile -t cpp:with_deps_camera .
```

**2. Run the container**, with GPU access, camera-device access, and this directory mounted at `/workspace`:
```bash
sudo docker run -it --name victor_projects_env \
  --user $(id -u):$(id -g) \
  --group-add $(getent group video | cut -d: -f3) \
  --device-cgroup-rule='c 81:* rmw' -v /dev:/dev -v /run/udev:/run/udev:ro \
  --shm-size 64G \
  --gpus 'all,"capabilities=compute,utility,graphics"' \
  -p 7861:7861 \
  --mount type=bind,source=/path/to/projects,target=/workspace,bind-propagation=shared \
  cpp:with_deps_camera
```
- `--gpus` exposes the NVIDIA GPU to the container.
- `--device-cgroup-rule` plus the two `-v` mounts expose `/dev/video*` to the container (needed only for `--camera` mode; skip if you only use `--input`/`--stream`).
- `--mount ...bind-propagation=shared` mounts your local `projects/` folder at `/workspace`, so files created or edited on either side stay in sync.

**3. Install stream-handling packages** (one-time, run inside the container since it isn't baked into the base image):
```bash
sudo apt-get update && sudo apt-get install -y \
  gstreamer1.0-plugins-good gstreamer1.0-libav gstreamer1.0-tools ffmpeg
```

**4. Build the C++ project:**
```bash
cd /workspace/2repo/yolov8m_cpp_onnx_camera_v8nano
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
Produces `build/yolov8n_camera_detector`, linked against the ONNX Runtime GPU build already vendored in `third_party/` and system OpenCV.

## Models

| File | Purpose | Compatibility |
|---|---|---|
| `models/best_singapore_2fps_1m_37m_fp16_640.onnx` | Driving detection (10 classes, `classes.txt`) | Supported |
| `models/best_singapore_fp32_640.onnx` | INT8 conversion source | Not supported by current runtime |
| `models/yolov8n_coco_fp16_640.onnx` | General-purpose validation (80 classes, `coco_classes.txt`) | Supported |

`models/classes.txt` (10 classes, in order, where class ID = line number, 0-indexed):
```
pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign
```

`models/coco_classes.txt` (80 standard COCO classes, same order Ultralytics ships).

## Model Input Resolution

This implementation is fixed to 640x640 input and 8400 predictions.

The included Singapore checkpoint was trained and exported at 640x640. Models exported at other resolutions are not compatible with the current hardcoded tensor dimensions in `PreprocessImageFP16`/`PostprocessOutputFP16` (`src/main.cpp`).

Use the project conversion scripts with the appropriate `imgsz` value for the target checkpoint.

## Usage

### CLI Reference

```
yolov8n_camera_detector --model <path> --classes <path> (--input <path> | --camera <index> | --stream <url-or-file>) [options]

Required:
  --model <path>        Path to FLOAT16-I/O yolov8n ONNX model (640x640 export)
  --classes <path>      Path to class names file (one per line)

Source (exactly one):
  --input <path>        Image file OR directory of images (batch mode)
  --camera <index>      Local camera device index (0 = /dev/video0, live mode)
  --stream <url>        RTSP URL OR a local video file path (live mode)

Detection options:
  --conf <float>        Confidence threshold (default 0.25)
  --nms <float>         NMS IoU threshold (default 0.5)
  --max-det <int>       Max detections per image/frame (default 300)

Output options (batch mode):
  --vis-dir/--labels-dir/--coco-out <path>, --vis/--no-vis, --yolo/--no-yolo, --coco/--no-coco

Live-mode options:
  --headless             Don't open a display window; console output only
  --save-video <path>    Record annotated output to a video file
  --metrics/--no-metrics Toggle per-frame FPS/timing overlay (default: on; needs --vis)
```

Run `./yolov8n_camera_detector --help` for the authoritative CLI reference.

### Reproducing the Singapore Driving Demo

The Singapore model is trained for road-scene imagery. Use representative driving footage for evaluation.

**1. Extract a short clip's worth of frames from a source video** (batch mode is more demo-friendly than live streaming, since each frame saves as it's processed, so you can stop anytime without losing output):

```bash
ffmpeg -ss 00:00:26 -i /path/to/source_video.mp4 -t 60 -vf fps=2 /path/to/output_frames/frame_%06d.jpg
```

Adjust `-ss` (start time) and `-t` (duration in seconds) for the segment you want.

**2. Run detection on the extracted frames:**

```bash
./yolov8n_camera_detector --model ../models/best_singapore_2fps_1m_37m_fp16_640.onnx --classes ../models/classes.txt --input /path/to/output_frames --conf 0.25
```

Annotated images land in `<output_frames>_vis/` (alongside `_yolo_labels/` and `_coco.json`, unless disabled).

### Live Camera or Stream

```bash
./yolov8n_camera_detector --model ../models/yolov8n_coco_fp16_640.onnx --classes ../models/coco_classes.txt --camera 0
```

This is the **COCO** model (general objects, matches an office/webcam scene). Swap in the Singapore model + `classes.txt` when the camera is pointed at driving/street footage instead.

## How It Works

The detection pipeline:

```
Input
  ↓
OpenCV capture / image loading
  ↓
Resize to 640x640
  ↓
FP16 preprocessing
  ↓
ONNX Runtime CUDA inference
  ↓
YOLOv8 output [1, 4+C, 8400]
  ↓
Confidence filtering + NMS
  ↓
Coordinate scaling
  ↓
Visualization / YOLO / COCO output
```

Preprocessing always resizes the incoming frame to 640x640 regardless of source resolution; postprocessing scales detection boxes back out using that frame's actual width/height, read fresh every call, so camera and video inputs at any resolution need no special handling.

## Roadmap: INT8 Quantization

INT8 inference is currently under development. `models/best_singapore_fp32_640.onnx` is retained as the source model for static quantization. The current C++ inference path supports FP16 tensors only, so INT8 integration will require an FP32-compatible tensor path for quantized ONNX models.

## Demo

| 10-class driving model: road footage | COCO 80-class model: live camera |
| :---: | :---: |
| ![10-class driving-model inference](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/cpp_onnx_camera/singapore_driving_demo.png) | ![COCO 80-class YOLOv8n live-camera inference](https://cdn.jsdelivr.net/gh/Victor-QTP/materials@main/cpp_onnx_camera/coco_live_camera.png) |