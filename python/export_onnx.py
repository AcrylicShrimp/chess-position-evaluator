"""
Export ValueOnlyModel to ONNX format.

Usage:
    cpe export-onnx <model-name>

Example:
    cpe export-onnx model-best
        artifacts/checkpoints/model-best.pth -> artifacts/onnx/model-best.onnx
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from libs.model import ValueOnlyModel
from libs.paths import ONNX_DIR, checkpoint_path, onnx_path


OPSET_VERSION = 20
EXAMPLE_BATCH_SIZE = 4
OUTPUT_ATOL = 1e-3


def run_export_onnx(model_name: str):
    """Export the given model to ONNX format."""
    input_path = checkpoint_path(model_name)
    output_path = onnx_path(model_name)

    # Check input exists
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Check output exists - prompt to overwrite
    if output_path.exists():
        response = input(f"{output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[1/4] Loading model from {input_path}")
    model = ValueOnlyModel()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Export to ONNX
    print(f"[2/4] Exporting to ONNX (opset {OPSET_VERSION})")
    dummy_input = torch.randn(EXAMPLE_BATCH_SIZE, 20, 8, 8)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["board"],
        output_names=["value"],
        dynamic_axes={
            "board": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        dynamo=False,
    )

    # Validate ONNX model structure
    print("[3/4] Validating ONNX model structure")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    # Validate outputs match
    print("[4/4] Validating outputs match (PyTorch vs ONNX Runtime)")
    test_input = torch.randn(4, 20, 8, 8)  # batch of 4

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(str(output_path))
    ort_output = session.run(None, {"board": test_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - ort_output))
    if max_diff > OUTPUT_ATOL:
        print(f"Warning: Max difference between outputs: {max_diff}")
    else:
        print(f"Outputs match (max diff: {max_diff:.2e})")

    print()
    print(f"Exported: {output_path}")
    print(f"Input:    board [batch, 20, 8, 8]")
    print(f"Output:   value [batch, 1]")
