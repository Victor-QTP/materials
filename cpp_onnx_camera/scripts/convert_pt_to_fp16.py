"""
Converts a YOLOv8 PyTorch checkpoint (.pt) DIRECTLY to a FLOAT16-I/O
ONNX model, in one script — combining the FP32 export step (which needs
PyTorch + ultralytics; there's no way around that, since onnxconverter_common
can only operate on already-exported ONNX files, never on a .pt checkpoint)
with the same onnxconverter_common-based FP16 conversion used by
convert_fp32_to_fp16.py (see that file, and ../docs/FP16_CONVERSION.md, for
why onnxconverter_common was chosen over ultralytics' own `half=True` export).

If you already have an FP32 .onnx file (e.g. exported previously, or by
someone else) and just need the FP16 conversion, use convert_fp32_to_fp16.py
directly instead — it's simpler and doesn't need PyTorch/ultralytics at all.
Reach for THIS script only when starting from a .pt checkpoint with no
existing ONNX export.

Prerequisites:
    pip install onnx onnxconverter_common
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    (ultralytics itself: used directly from a local clone via sys.path below,
    not pip-installed — see ../../ultralytics_repo/)

Usage:
    python3 convert_pt_to_fp16.py <input.pt> <output_fp16.onnx> [imgsz]

<imgsz> defaults to 640 in this copy of the script (this folder is for
yolov8n models trained at 640x640 — see args.yaml of the training run).
Pass the resolution the .pt was actually TRAINED at, not an arbitrary
value: exporting at a different resolution than training silently degrades
detection quality/confidence instead of raising an error.

===============================================================================
WHY THE TEMP-DIRECTORY DANCE BELOW
===============================================================================
ultralytics' `model.export()` does not take an explicit output path — it
always saves next to the source .pt file, using the same base filename with
a ".onnx" extension (e.g. "best.pt" -> "best.onnx" in the SAME directory).

During this project's development, exactly this behavior caused a real
incident: running an export against a .pt file that lived in the same
directory as an existing, working "best.onnx" silently OVERWROTE that
existing file, because nothing told ultralytics to save anywhere else (see
../docs/FP16_CONVERSION.md for the full story). The file was recovered from
a backup, but the lesson stuck: never let this export run in a directory
that might contain files you care about.

This script avoids that entirely by copying the input .pt into a fresh
temporary directory first (guaranteed empty, guaranteed to contain nothing
of value), running the export there, and reading the result back into memory
before the temp directory (and everything in it, including the intermediate
FP32 .onnx) is automatically deleted. The ONLY file this script writes
outside that temp directory is the one path you explicitly pass as
<output_fp16.onnx> — nothing else on disk is ever touched.
"""
import os
import shutil
import sys
import tempfile

import onnx
from onnxconverter_common import float16

# Reuses the Cast-node-type-mismatch fix from convert_fp32_to_fp16.py rather
# than duplicating it — both scripts hit the exact same onnxconverter_common
# bug, so the fix lives in one place. Requires running this script from
# inside the scripts/ directory (or otherwise having scripts/ on sys.path)
# so this import can find the sibling file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_fp32_to_fp16 import fix_cast_output_types

# Path to the local ultralytics clone used throughout this project — NOT a
# pip-installed package. See ../docs/FP16_CONVERSION.md for why (avoids a
# multi-GB dependency chain beyond what PyTorch itself already needs).
ULTRALYTICS_REPO_PATH = "/workspace/mycppcode/projects/ultralytics_repo"


def export_fp32_onnx_isolated(pt_path: str, imgsz: int) -> onnx.ModelProto:
    """Exports a .pt checkpoint to FP32 ONNX inside a throwaway temp
    directory, returns the loaded ONNX model, and cleans up all temp files
    (including the intermediate .onnx) before returning. See the module
    docstring above for why this isolation matters.
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
        # Deliberately NOT passing half=True here: this project's FP16
        # conversion goes through onnxconverter_common (below), not
        # ultralytics' own native half=True path — see
        # ../docs/FP16_CONVERSION.md for the reasoning. This call produces
        # a plain FP32 export, matching the original best.onnx exactly.
        exported_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
        print(f"FP32 export (temporary): {exported_path}")

        # Read the exported file into memory NOW, while still inside the
        # "with" block — everything in tmp_dir, including exported_path
        # itself, is deleted the moment this block exits.
        fp32_model = onnx.load(exported_path)

    return fp32_model


def main():
    if len(sys.argv) not in (3, 4):
        print(f"Usage: {sys.argv[0]} <input.pt> <output_fp16.onnx> [imgsz=640]")
        sys.exit(1)

    pt_path, output_path = sys.argv[1], sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) == 4 else 640

    fp32_model = export_fp32_onnx_isolated(pt_path, imgsz)

    print("Converting to FLOAT16 (onnxconverter_common, default op_block_list, "
          "which already includes Resize)...")
    fp16_model = float16.convert_float_to_float16(fp32_model)

    print("Patching Cast node output type declarations...")
    num_fixed = fix_cast_output_types(fp16_model)
    print(f"Fixed {num_fixed} node(s)")

    print("Validating with onnx.checker...")
    onnx.checker.check_model(fp16_model)
    print("onnx.checker.check_model: PASSED")

    onnx.save(fp16_model, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
