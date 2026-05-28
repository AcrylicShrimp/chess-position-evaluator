import importlib
import sys
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

model_module = importlib.import_module("libs.model")
registry_module = importlib.import_module("libs.modeling.registry")


class ModelRegistryTest(unittest.TestCase):
    def test_registry_ids_match_supported_model_variants(self):
        self.assertEqual(
            tuple(registry_module.MODEL_VARIANTS.keys()),
            model_module.SUPPORTED_MODEL_VARIANTS,
        )

    def test_registry_specs_match_constructed_parameter_counts(self):
        for variant_id, spec in registry_module.MODEL_VARIANTS.items():
            with self.subTest(variant_id=variant_id):
                model = model_module.ValueOnlyModel(model_variant=variant_id)
                parameter_count = sum(p.numel() for p in model.parameters())

                self.assertEqual(spec.id, variant_id)
                self.assertEqual(spec.expected_params, parameter_count)

    def test_unknown_variant_fails_through_registry(self):
        with self.assertRaisesRegex(ValueError, "unsupported model variant"):
            registry_module.get_model_variant_spec("unknown")

        with self.assertRaisesRegex(ValueError, "unsupported model variant"):
            model_module.ValueOnlyModel(model_variant="unknown")

    def test_benchmark_checkpoint_specs_reference_known_variants(self):
        checkpoint_names = registry_module.default_benchmark_model_names()

        self.assertEqual(
            len(checkpoint_names),
            len(registry_module.DEFAULT_BENCHMARK_CHECKPOINTS),
        )
        for checkpoint in registry_module.DEFAULT_BENCHMARK_CHECKPOINTS:
            with self.subTest(checkpoint=checkpoint.checkpoint_name):
                self.assertIn(
                    checkpoint.variant_id,
                    registry_module.MODEL_VARIANTS,
                )
                self.assertTrue(checkpoint.checkpoint_name)
                self.assertTrue(checkpoint.role)

    def test_checkpoint_variant_default_is_unchanged(self):
        self.assertEqual(
            model_module.model_variant_from_checkpoint({}),
            model_module.MODEL_VARIANT_STACKED_EDGE_GATE_FFN,
        )


if __name__ == "__main__":
    unittest.main()
