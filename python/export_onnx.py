"""
Export EvalOnlyModel to ONNX format.

Usage:
    cpe export-onnx <model-name>

Example:
    cpe export-onnx model-best
        models/checkpoints/model-best.pth -> models/onnx/model-best.onnx
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.export import Dim

from libs.model import EvalOnlyModel


CHECKPOINTS_DIR = "models/checkpoints"
ONNX_DIR = "models/onnx"
OPSET_VERSION = 21
EXAMPLE_BATCH_SIZE = 2 * 1024


def run_export_onnx(model_name: str):
    """Export the given model to ONNX format."""
    input_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}.pth")
    output_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")

    # Check input exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Check output exists - prompt to overwrite
    if os.path.exists(output_path):
        response = input(f"{output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    os.makedirs(ONNX_DIR, exist_ok=True)

    # Load model
    print(f"[1/4] Loading model from {input_path}")
    model = EvalOnlyModel()
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Export to ONNX
    print(f"[2/4] Exporting to ONNX (opset {OPSET_VERSION})")
    dummy_input = torch.randn(EXAMPLE_BATCH_SIZE, 18, 8, 8)

    # For dynamo exporter: dynamic_shapes keys must match forward() arg names
    # EvalOnlyModel.forward(self, x) -> arg name is "x"
    batch_dim = Dim("batch", min=1)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["board"],
        output_names=["value"],
        dynamic_shapes={"x": {0: batch_dim}},
        opset_version=OPSET_VERSION,
        dynamo=True,
        optimize=True,
    )

    # Validate ONNX model structure
    print("[3/4] Validating ONNX model structure")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    # Validate outputs match
    print("[4/4] Validating outputs match (PyTorch vs ONNX Runtime)")
    test_input = torch.randn(4, 18, 8, 8)  # batch of 4

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    session = ort.InferenceSession(output_path)
    ort_output = session.run(None, {"board": test_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - ort_output))
    if max_diff > 1e-5:
        print(f"Warning: Max difference between outputs: {max_diff}")
    else:
        print(f"Outputs match (max diff: {max_diff:.2e})")

    print()
    print(f"Exported: {output_path}")
    print(f"Input:    board [batch, 18, 8, 8]")
    print(f"Output:   value [batch, 1]")
