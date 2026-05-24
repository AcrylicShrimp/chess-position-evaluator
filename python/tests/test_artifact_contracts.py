import sys
import unittest
from pathlib import Path

import onnx

PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PYTHON_ROOT.parent
sys.path.insert(0, str(PYTHON_ROOT))


def onnx_input_shape(path: Path) -> list[int | str | None]:
    model = onnx.load(path, load_external_data=False)
    graph_input = model.graph.input[0]
    dims = graph_input.type.tensor_type.shape.dim

    shape: list[int | str | None] = []
    for dim in dims:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


class ArtifactContractTest(unittest.TestCase):
    def test_at_least_one_committed_onnx_artifact_uses_current_input_shape(self):
        onnx_paths = sorted((REPO_ROOT / "models/onnx").glob("*.onnx"))
        self.assertTrue(onnx_paths, "No committed ONNX artifacts found")

        compatible_paths = [
            path for path in onnx_paths if onnx_input_shape(path)[1:] == [20, 8, 8]
        ]

        self.assertTrue(
            compatible_paths,
            "No committed ONNX artifacts declare input shape [batch, 20, 8, 8]",
        )


if __name__ == "__main__":
    unittest.main()
