from __future__ import annotations

import unittest

from mlxr.families.ltx._checkpoint_contract import (
    resolve_transformer_semantic_contract,
)


class TransformerCheckpointContractTests(unittest.TestCase):
    def test_prompt_contract_rejects_partial_v2_metadata(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "Partial V2 LTX prompt config"
        ):
            resolve_transformer_semantic_contract(
                transformer_config={
                    "rope_type": "split",
                    "frequencies_precision": "float64",
                    "caption_proj_before_connector": True,
                    "caption_projection_first_linear": False,
                    "connector_apply_gated_attention": True,
                },
                source="checkpoint.safetensors",
                has_v2_weights=False,
            )

    def test_generation_contract_allows_partial_prompt_layout_metadata(self) -> None:
        contract = resolve_transformer_semantic_contract(
            transformer_config={
                "rope_type": "split",
                "frequencies_precision": "float64",
                "caption_proj_before_connector": True,
            },
            source="checkpoint.safetensors",
            fallback_apply_gated_attention=True,
            fallback_cross_attention_adaln=True,
            enforce_prompt_v2_layout=False,
        )

        self.assertEqual(contract.rope_type, "split")
        self.assertTrue(contract.double_precision_rope)
        self.assertTrue(contract.caption_proj_before_connector)
        self.assertTrue(contract.apply_gated_attention)
        self.assertTrue(contract.cross_attention_adaln)
        self.assertEqual(contract.av_ca_timestep_scale_multiplier, 1)


if __name__ == "__main__":
    unittest.main()
