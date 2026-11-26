import hashlib
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EXPORTER = (
    Path(__file__).resolve().parents[4]
    / "examples/ling31_dspark_quantization/quantize_draft.py"
)
EXPORT = runpy.run_path(str(EXPORTER))


class TestLingDSparkInt4Export(unittest.TestCase):
    def test_quantization_scope(self):
        policy = EXPORT["should_quantize"]
        matrix = SimpleNamespace(ndim=2)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            self.assertTrue(
                policy(f"dspark.layers.0.mlp.experts.0.{projection}.weight", matrix)
            )
        for name in (
            "layers.0.mlp.experts.0.gate_proj.weight",
            "dspark.layers.0.mlp.shared_experts.gate_proj.weight",
            "dspark.layers.0.mlp.gate.weight",
            "dspark.layers.0.attention.kv_b_proj.weight",
            "dspark.target_proj.weight",
            "dspark.markov_head.weight",
            "dspark.confidence_head.proj.weight",
        ):
            self.assertFalse(policy(name, matrix), name)
        self.assertFalse(
            policy(
                "dspark.layers.0.mlp.experts.0.gate_proj.weight",
                SimpleNamespace(ndim=1),
            )
        )

    def test_weight_only_config_disables_grouped_activation_scales(self):
        from humming.layer import HummingInputSchema
        from humming.schema.humming import HummingWeightSchema

        config = EXPORT["INT4_CONFIG"]
        weight = HummingWeightSchema.from_config(config)
        activation = HummingInputSchema.from_config(config)
        self.assertEqual(weight.b_dtype.num_bits, 4)
        self.assertEqual(weight.weight_scale_group_size, 64)
        self.assertEqual(activation.a_dtype.num_bits, 16)
        self.assertEqual(activation.input_scale_group_size, 0)
        self.assertEqual(config["ignored_layers"], ["attention", "shared_experts"])

    def test_target_activation_override_does_not_requantize_draft(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.humming import HummingConfig

        config = HummingConfig.from_config(EXPORT["INT4_CONFIG"])
        override = {
            "dtype": "float8e4m3",
            "group_size": 32,
            "ignored_layers": [r"re:^layers\."],
        }
        with patch.object(
            envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG, "get", return_value=override
        ):
            draft = config.get_quant_config_for_layer("layers.0.mlp.experts", "moe")
            target = config.get_quant_config_for_layer(
                "model.layers.0.mlp.experts", "moe"
            )
        self.assertEqual(draft.input_schema.a_dtype.num_bits, 16)
        self.assertIsNone(draft.force_input_schema)
        self.assertEqual(target.force_input_schema.a_dtype.num_bits, 8)

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA quantization kernels"
    )
    def test_gpu_export_preserves_nonexpert_tensors_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "output"
            source.mkdir()
            tensors = {
                "dspark.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(
                    128, 256
                ).bfloat16(),
                "dspark.layers.0.mlp.experts.0.up_proj.weight": torch.randn(
                    128, 256
                ).bfloat16(),
                "dspark.layers.0.mlp.experts.0.down_proj.weight": torch.randn(
                    256, 128
                ).bfloat16(),
                "dspark.layers.0.attention.kv_b_proj.weight": torch.randn(
                    128, 256
                ).bfloat16(),
                "dspark.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(
                    128, 256
                ).bfloat16(),
                "dspark.target_proj.weight": torch.randn(32, 32).bfloat16(),
            }
            weights = source / "draft_model.safetensors"
            save_file(tensors, str(weights))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            config = {
                "dspark_bundle_format": "namespaced_safetensors",
                "dspark_bundle_version": 1,
                "dspark_config": {"architectures": ["LingV3ForCausalLMDSparkMLA"]},
            }
            (source / "config.json").write_text(json.dumps(config))
            command = [
                sys.executable,
                str(EXPORTER),
                "--source",
                str(source),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            exported = {}
            for shard in set(index["weight_map"].values()):
                exported.update(load_file(str(output / shard)))
            for name, tensor in tensors.items():
                if ".mlp.experts." in name:
                    self.assertEqual(exported[name].dtype, torch.int32)
                    self.assertEqual(
                        exported[name].shape, (tensor.shape[0], tensor.shape[1] // 8)
                    )
                    self.assertIn(
                        name.removesuffix("weight") + "weight_scale", exported
                    )
                else:
                    self.assertTrue(torch.equal(exported[name], tensor))
            self.assertEqual(hashlib.sha256(weights.read_bytes()).hexdigest(), digest)
            out_config = json.loads((output / "config.json").read_text())
            self.assertEqual(
                out_config["quantization_config"],
                out_config["dspark_config"]["quantization_config"],
            )
            self.assertEqual(
                out_config["quantization_config"]["input_scale_group_size"], 0
            )
            self.assertNotEqual(
                subprocess.run(command, capture_output=True, check=False).returncode, 0
            )


if __name__ == "__main__":
    unittest.main()
