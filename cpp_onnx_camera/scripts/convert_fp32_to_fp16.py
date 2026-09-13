"""
Converts an FP32 YOLOv8 ONNX model to FLOAT16 input/output, using
only `onnx` + `onnxconverter_common` (pure-Python, lightweight) — no
PyTorch, no ultralytics. This is the reproducible recipe behind
models/best_fp16.onnx in this project.

Prerequisite: an existing FP32 ONNX export (e.g. produced once via
ultralytics' `model.export(format="onnx", imgsz=1280, simplify=True,
opset=17)` elsewhere, or any other standard YOLOv8 ONNX export). This script
does NOT do that export step — it only takes an already-exported FP32 .onnx
file and converts it.

Install:
    pip install onnx onnxconverter_common

Usage:
    python3 convert_fp32_to_fp16.py <input_fp32.onnx> <output_fp16.onnx>

===============================================================================
WHAT THIS SCRIPT ACTUALLY FIXES (read this before touching a new model)
===============================================================================

An ONNX file is a serialized "graph": a list of NODES (operations like Cast,
Resize, Conv — each with inputs, outputs, and attributes) plus a separate
VALUE_INFO table that independently declares the data type and shape of every
named tensor flowing between those nodes. These two things — what a node's
own definition says it produces, and what the value_info table separately
claims that same tensor's type is — are supposed to always agree. When they
don't, ONNX Runtime's loader refuses to load the model at all.

`onnxconverter_common.float16.convert_float_to_float16()` converts an FP32
model to FP16. Some ops (Resize, NonMaxSuppression, TopK, and others — see
its DEFAULT_OP_BLOCK_LIST) are deliberately left in FP32, because converting
them to FP16 would be unsafe or unsupported. To keep a blocked FP32 op
plugged into an otherwise-FP16 graph, the converter auto-inserts a Cast node
right after it, converting that op's FP32 output to FP16 before it's used
downstream.

The bug: when inserting that Cast node, the converter correctly sets the
Cast node's own "to" attribute to FLOAT16 (the node itself is fine), but for
some Cast nodes it fails to also update the graph's separate value_info
entry for that same output tensor — leaving value_info still saying FLOAT.
Result: the Cast node says "I output FLOAT16," the graph's own metadata for
that exact tensor says "no, this is FLOAT" — a direct contradiction. ONNX
Runtime's loader checks this and rejects the model with an error like:

    Type Error: Type (tensor(float)) of output arg (.../Resize_output_0)
    of node (.../Resize_output_cast0) does not match expected type
    (tensor(float16)).

Note the node name ends in "_cast0" — that's this bug's fingerprint. The
node upstream of it (here, a Resize) isn't actually broken; it's the
auto-generated Cast node's bookkeeping that's inconsistent.

THE FIX (function fix_cast_output_types, below): don't try to figure out
which specific nodes are broken by name — just walk EVERY Cast node in the
graph, and for each one, force the value_info entry for its output to match
its own "to" attribute. This is always safe and always correct, because a
Cast node's own "to" attribute is the authoritative truth for what type it
produces — value_info must agree with the node, never the other way around.
Any Cast node that was already consistent gets left untouched (no-op); any
that had this bug gets corrected.

===============================================================================
HOW TO REUSE THIS FOR A DIFFERENT MODEL OR A DIFFERENT VERSION OF THIS BUG
===============================================================================

1. Run this script's two steps (convert, then fix_cast_output_types) on your
   FP32 model, same as below.
2. Try loading the result in ONNX Runtime. If it loads: done.
3. If it still fails with a *different* "Type (tensor(X)) ... does not match
   expected type (tensor(Y))" error, the same underlying idea generalizes:
   the error message names the specific node and tensor that disagree with
   each other. fix_cast_output_types already handles every Cast node
   unconditionally, so if the failing node is a Cast node, this script
   should already have fixed it — check whether onnx.checker.check_model()
   actually passed (this script calls it and will raise if not; don't
   comment that out). If the failing node is NOT a Cast node (e.g. some
   other op type entirely), the same general technique still applies, but
   you'd need to adapt fix_cast_output_types' node.op_type filter (currently
   `if node.op_type != "Cast": continue`) to also handle that op type — the
   rest of the logic (compare declared value_info against what the node
   itself claims to produce, overwrite to match) carries over unchanged.
4. If onnx.checker.check_model() itself fails (not just ONNX Runtime), the
   problem is more fundamental than this specific value_info mismatch — this
   script's fix won't be sufficient, and the graph needs deeper inspection
   (e.g. print every node's op_type and attributes with a short script using
   onnx.helper.printable_graph(model.graph) to see the whole structure).
"""
import sys
import onnx
from onnxconverter_common import float16


def fix_cast_output_types(model: onnx.ModelProto) -> int:
    """Patches value_info type mismatches on auto-inserted Cast node outputs.

    Walks every node in the graph; for each one that is a Cast op, reads
    what type it declares it casts TO (its "to" attribute — the ground
    truth), then checks the graph's separately-stored value_info entry for
    that same output tensor name. If value_info disagrees, overwrites it to
    match the Cast node. See the module docstring above for the full "why".

    Returns the number of fixes applied (0 means the graph was already
    consistent — not necessarily an error, just means this particular bug
    wasn't present).
    """
    # model.graph is the ONNX GraphProto — the actual computational graph:
    # a list of nodes (operations), plus separate lists describing the
    # graph's inputs, outputs, and "value_info" (type/shape metadata for
    # every other intermediate tensor that isn't an input or output).
    graph = model.graph

    # Build a lookup dictionary: tensor name -> its ValueInfoProto (the
    # protobuf object holding that tensor's declared type/shape). We need
    # this because a Cast node only stores its OUTPUT'S NAME (a string) —
    # to find and edit that tensor's type declaration, we have to search for
    # the matching entry by name. Building this dict once up front (instead
    # of re-scanning graph.value_info inside the loop below for every single
    # node) turns what would be an O(nodes x value_infos) search into a fast
    # O(1) dictionary lookup per node.
    #
    # A tensor can have its type declared in one of two places depending on
    # whether it's a purely internal/intermediate tensor (graph.value_info)
    # or one of the graph's actual declared outputs (graph.output) — we
    # check both, since we don't know in advance which one a given Cast
    # node's output might be.
    value_info_by_name = {}
    for vi in graph.value_info:
        value_info_by_name[vi.name] = vi
    for vi in graph.output:
        value_info_by_name[vi.name] = vi

    fixed = 0
    for node in graph.node:
        # Every node has an "op_type" string naming what operation it is
        # (e.g. "Conv", "Resize", "Cast", "Relu", ...). We only care about
        # Cast nodes here — everything else is left completely untouched.
        if node.op_type != "Cast":
            continue

        # A node's "attribute" list holds its configuration values (op-type
        # specific — Conv has stride/padding attributes, Cast has a single
        # "to" attribute naming the type it converts to). We scan for the
        # one literally named "to". "attr.i" is the integer field of that
        # attribute's value — ONNX's type enum (FLOAT=1, FLOAT16=10, etc.,
        # the same numbering used everywhere in this project, e.g. the
        # elem_type checks in src/main.cpp).
        to_type = None
        for attr in node.attribute:
            if attr.name == "to":
                to_type = attr.i
        if to_type is None:
            continue  # malformed Cast node with no "to" attribute at all; skip

        # A node can (rarely) have multiple outputs; Cast always has exactly
        # one, but looping over node.output handles it generally regardless.
        for output_name in node.output:
            vi = value_info_by_name.get(output_name)
            if vi is None:
                # This Cast node's output isn't in our lookup table at all —
                # e.g. it might not be recorded anywhere in value_info/output
                # (can happen for certain intermediate tensors). Nothing to
                # patch in that case; ONNX Runtime derives its own type
                # information for untracked tensors instead of trusting a
                # (nonexistent) declaration.
                continue

            # THE ACTUAL BUG CHECK: does the graph's own stored type
            # declaration for this tensor (vi.type.tensor_type.elem_type)
            # disagree with what the Cast node itself says it produces
            # (to_type)? If so, that's exactly the inconsistency ONNX
            # Runtime's loader rejects.
            if vi.type.tensor_type.elem_type != to_type:
                print(f"Fixing {node.name}: output '{output_name}' declared "
                      f"elem_type={vi.type.tensor_type.elem_type}, "
                      f"node casts to={to_type}")
                # THE ACTUAL FIX: overwrite the graph's stored declaration to
                # match the Cast node's own, authoritative "to" attribute.
                # This directly mutates the ValueInfoProto object in place
                # (protobuf message fields are mutable like this) — since
                # `vi` came from our dictionary, which points at the same
                # underlying objects living inside `graph`, this edit is
                # visible in `model` itself; no need to write it back
                # separately.
                vi.type.tensor_type.elem_type = to_type
                fixed += 1
    return fixed


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_fp32.onnx> <output_fp16.onnx>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    print(f"Loading FP32 model: {input_path}")
    # onnx.load reads the .onnx file (a serialized protobuf) from disk and
    # parses it into an in-memory onnx.ModelProto object we can inspect and
    # modify with Python — this is the standard entry point for any
    # programmatic ONNX graph manipulation, not specific to this script.
    model = onnx.load(input_path)

    print("Converting to FLOAT16 (default op_block_list, which already "
          "includes Resize)...")
    # Calling this with NO extra arguments (no op_block_list=...) uses the
    # library's own DEFAULT_OP_BLOCK_LIST, which already includes "Resize"
    # (and NonMaxSuppression, TopK, Range, Min, Max, Upsample, and others) —
    # i.e. this call alone does NOT need any YOLO-specific configuration to
    # correctly keep Resize in FP32; that part of the conversion was never
    # the bug. Only the Cast-node bookkeeping around those blocked ops was
    # broken, which is what fix_cast_output_types patches below.
    model_fp16 = float16.convert_float_to_float16(model)

    print("Patching Cast node output type declarations...")
    num_fixed = fix_cast_output_types(model_fp16)
    print(f"Fixed {num_fixed} node(s)")

    print("Validating with onnx.checker...")
    # onnx.checker.check_model performs ONNX's own structural/schema
    # validation of the graph — including exactly the kind of node-output-
    # type-vs-value_info consistency check this script's fix addresses. It
    # RAISES an exception if the graph is still invalid, which is
    # deliberately NOT caught here: if this line fails, the model is not
    # safe to save/use, and the script should crash loudly rather than
    # silently write out a broken file.
    onnx.checker.check_model(model_fp16)
    print("onnx.checker.check_model: PASSED")

    # onnx.save serializes the (now-patched) in-memory model back out to a
    # .onnx file — the mirror operation of onnx.load above.
    onnx.save(model_fp16, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # SYNTAX: this guard means the code inside only runs when this file is
    # executed directly as a script (e.g. "python3 convert_fp32_to_fp16.py
    # ..."), NOT when it's imported as a module from other Python code
    # (e.g. "from convert_fp32_to_fp16 import fix_cast_output_types" from a
    # different script) — the standard Python convention for a file that's
    # meant to be usable both ways.
    main()
