"""
Converts a YOLOv8 PyTorch checkpoint (.pt) to a plain FP32 ONNX export and
saves it to disk — no FP16 conversion, no quantization. This is
convert_pt_to_fp16.py with the FP16 conversion step removed: same isolated
temp-directory export (see that file's docstring for why the temp-directory
dance matters — avoids ultralytics silently overwriting an existing .onnx
next to the source .pt), but the exported FP32 model is what actually gets
saved, instead of being converted-then-discarded.

Reach for THIS script when you need a real, persisted FP32 .onnx file as a
starting point for something other than convert_fp32_to_fp16.py — e.g. INT8
static quantization via onnxruntime.quantization.quantize_static(), which
operates on FP32 graphs, not already-FP16 ones.

Prerequisites:
    pip install onnx
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    (ultralytics itself: used directly from a local clone via sys.path below,
    not pip-installed — see ../../ultralytics_repo/)

Usage:
    python3 convert_pt_to_fp32.py <input.pt> <output_fp32.onnx> [imgsz]

<imgsz> defaults to 640 in this copy of the script (this folder is for
yolov8n models trained at 640x640 — see args.yaml of the training run).
Pass the resolution the .pt was actually TRAINED at, not an arbitrary
value: exporting at a different resolution than training silently degrades
detection quality/confidence instead of raising an error (see
convert_pt_to_fp16.py's docstring for the real incident this bit us on).
"""
import os
import shutil
import sys
import tempfile

import onnx

# Path to the local ultralytics clone used throughout this project — NOT a
# pip-installed package. See ../docs/FP16_CONVERSION.md for why (avoids a
# multi-GB dependency chain beyond what PyTorch itself already needs).
ULTRALYTICS_REPO_PATH = "/workspace/mycppcode/projects/ultralytics_repo"


def export_fp32_onnx_isolated(pt_path: str, imgsz: int) -> onnx.ModelProto:
    """Exports a .pt checkpoint to FP32 ONNX inside a throwaway temp
    directory, returns the loaded ONNX model, and cleans up all temp files
    before returning. Identical to the same-named function in
    convert_pt_to_fp16.py — duplicated here rather than imported so this
    script has no dependency on that one (and vice versa).
    """
    sys.path.insert(0, ULTRALYTICS_REPO_PATH)
    from ultralytics import YOLO  # imported here, not at module level, so

    # that this script's other functions (e.g. if imported elsewhere) don't
    # require ultralytics/torch to be importable just to be defined.

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy the .pt into the temp directory rather than loading it
        # in-place — this is what guarantees ultralytics' own
        # same-directory default output naming lands somewhere disposable,
        # never next to the original .pt file.
        tmp_pt_path = os.path.join(tmp_dir, os.path.basename(pt_path))
        shutil.copy(pt_path, tmp_pt_path)

        print(f"Loading PyTorch weights (isolated copy): {tmp_pt_path}")
        model = YOLO(tmp_pt_path)

        print(f"Exporting FP32 ONNX (imgsz={imgsz}, opset=17, simplify=True)...")
        exported_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
        print(f"FP32 export (temporary): {exported_path}")

        # Read the exported file into memory NOW, while still inside the
        # "with" block — everything in tmp_dir, including exported_path
        # itself, is deleted the moment this block exits.
        fp32_model = onnx.load(exported_path)

    return fp32_model


def main():
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <input.pt> <output_fp32.onnx> [imgsz=640]")
        sys.exit(1)

    pt_path, output_path = sys.argv[1], sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) == 4 else 640

    fp32_model = export_fp32_onnx_isolated(pt_path, imgsz)

    print("Validating with onnx.checker...")
    onnx.checker.check_model(fp32_model)
    print("onnx.checker.check_model: PASSED")

    onnx.save(fp32_model, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
