import copy
import unittest
from types import SimpleNamespace

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.speculative.dspark_components.dspark_config import (
    apply_namespaced_dspark_draft_config,
    checkpoint_bundles_dspark_draft,
    is_namespaced_dspark_bundle,
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def bundle():
    return SimpleNamespace(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        use_kda=True,
        num_hidden_layers=72,
        num_nextn_predict_layers=0,
        dspark_bundle_format="namespaced_safetensors",
        dspark_bundle_version=1,
        dspark_config={
            "architectures": ["LingV3ForCausalLMDSparkMLA"],
            "attention_type": "mla",
            "num_hidden_layers": 3,
            "block_size": 5,
            "target_layer_ids": [69, 70, 71],
        },
    )


class TestLingDSparkBundle(unittest.TestCase):
    def test_bundle_detection_and_projection(self):
        target = bundle()
        draft = copy.deepcopy(target)
        self.assertTrue(checkpoint_bundles_dspark_draft(target))
        self.assertTrue(apply_namespaced_dspark_draft_config(draft))
        self.assertEqual(draft.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(draft.num_hidden_layers, 3)
        self.assertEqual(draft.full_attention_type, "mla")
        self.assertEqual(draft.attention_type, 1)
        self.assertFalse(draft.use_kda)
        self.assertTrue(target.use_kda)
        self.assertEqual(target.num_hidden_layers, 72)
        self.assertEqual(target.architectures, ["BailingMoeV3ForCausalLM"])

    def test_model_config_does_not_remap_bundle_to_nextn(self):
        config = bundle()
        runner = SimpleNamespace(
            is_draft_model=True,
            speculative_algorithm="DSPARK",
            hf_config=config,
            hf_text_config=config,
        )
        ModelConfig._config_draft_model(runner)
        self.assertEqual(config.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(config.num_hidden_layers, 3)

    def test_target_config_is_not_modified(self):
        config = bundle()
        runner = SimpleNamespace(
            is_draft_model=False,
            speculative_algorithm="DSPARK",
            hf_config=config,
            hf_text_config=config,
        )
        ModelConfig._config_draft_model(runner)
        self.assertEqual(config.architectures, ["BailingMoeV3ForCausalLM"])
        self.assertEqual(config.num_hidden_layers, 72)

    def test_unsupported_version_is_not_projected(self):
        config = bundle()
        config.dspark_bundle_version = 2
        self.assertFalse(is_namespaced_dspark_bundle(config))
        self.assertFalse(apply_namespaced_dspark_draft_config(config))
        self.assertEqual(config.num_hidden_layers, 72)

    def test_invalid_architecture_fails_before_mutation(self):
        config = bundle()
        config.dspark_config["architectures"] = []
        with self.assertRaises(ValueError):
            apply_namespaced_dspark_draft_config(config)
        self.assertEqual(config.num_hidden_layers, 72)

    def test_parser_uses_nested_gamma_and_mla_without_mutation(self):
        config = bundle()
        parsed = parse_dspark_draft_config(draft_hf_config=config)
        self.assertEqual(parsed.gamma, 5)
        self.assertEqual(parsed.num_hidden_layers, 3)
        self.assertTrue(parsed.is_mla)
        self.assertEqual(parsed.target_layer_ids, [69, 70, 71])
        self.assertEqual(config.num_hidden_layers, 72)

    def test_native_mla_draft_does_not_get_a_linear_backend(self):
        config = bundle()
        apply_namespaced_dspark_draft_config(config)
        self.assertIsNone(mambaish_config(SimpleNamespace(hf_config=config)))

    def test_legacy_bundle_still_detected(self):
        config = SimpleNamespace(dspark_block_size=5)
        self.assertTrue(checkpoint_bundles_dspark_draft(config))
        self.assertFalse(apply_namespaced_dspark_draft_config(config))


if __name__ == "__main__":
    unittest.main()
