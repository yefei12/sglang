"""Ling v3 MLA/MoE draft for DSpark speculative decoding."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.factories import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoELinearDecoderLayer,
    prepare_bailing_mla_weights,
)
from sglang.srt.models.dspark import (
    build_confidence_head,
    build_markov_head,
    gather_and_crop_vocab,
    project_through_lm_head,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
    is_namespaced_dspark_bundle,
)
from sglang.srt.utils import BumpAllocator
from torch import nn

logger = logging.getLogger(__name__)


class LingV3ForCausalLMDSparkMLA(nn.Module):
    """Ling DSpark draft using the native MLA parameterization."""

    supports_pre_gather_target_hidden_projection = False

    _attention_output_alias = ".attention.o_proj."

    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.is_bundled = is_namespaced_dspark_bundle(config)
        self.num_context_features = len(config.target_layer_ids)
        self.block_size = int(config.block_size)
        self.gamma = self.block_size
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        self.num_stages = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.target_proj = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.target_norm = RMSNorm(hidden_size, eps=eps)
        with layer_stack():
            self.layers = nn.ModuleList(
                [
                    BailingMoELinearDecoderLayer(
                        config=config,
                        quant_config=quant_config,
                        layer_id=layer_id,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                    for layer_id in range(int(config.num_hidden_layers))
                ]
            )
        for layer in self.layers:
            layer.attention.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            layer.attention.attn_mha.attn_type = AttentionType.ENCODER_ONLY
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.embed_tokens: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError(
                "Ling DSpark draft requires target embeddings to be attached."
            )
        return self.embed_tokens(input_ids)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self.num_context_features * int(self.config.hidden_size)
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "Ling DSpark target hidden shape mismatch: expected "
                f"[N, {expected}], got {tuple(target_hidden.shape)}."
            )
        return self.target_norm(self.target_proj(target_hidden))

    @torch.inference_mode()
    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d=None,
        commit_lens=None,
        target_hidden_is_projected: bool = False,
        swa_loc=None,
    ) -> None:
        if target_hidden_is_projected or swa_loc is not None:
            raise ValueError(
                "Ling MLA draft expects unprojected hidden states and full KV locations."
            )
        if (cache_loc_2d is None) != (commit_lens is None):
            raise ValueError(
                "Ling DSpark MLA requires cache_loc_2d and commit_lens together."
            )
        if cache_loc_2d is not None:
            verify_len = cache_loc_2d.shape[1]
            col = torch.arange(verify_len, device=cache_loc_2d.device).view(1, -1)
            committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            target_hidden = target_hidden[committed]
            positions = positions[committed]
            cache_loc = cache_loc_2d.reshape(-1)[committed]
            if target_hidden.numel() == 0:
                return
        self._write_target_hidden_kv_rows(
            target_hidden=target_hidden,
            pool=pool,
            positions=positions,
            cache_loc=cache_loc,
        )

    def _write_target_hidden_kv_rows(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        context = self.project_target_hidden(target_hidden)
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
                # q_lora_rank fuses the query and KV down-projections into one
                # GEMM; the KV latent sits behind the leading q_lora_rank slice.
                fused, _ = attn.fused_qkv_a_proj_with_mqa(context)
                latent = fused[..., attn.q_lora_rank :]
            else:
                latent, _ = attn.kv_a_proj_with_mqa(context)
            k_nope = attn.kv_a_layernorm(latent[..., : attn.kv_lora_rank]).unsqueeze(1)
            k_rope = latent[..., attn.kv_lora_rank :].unsqueeze(1)
            dummy_q = k_rope.new_empty(
                k_rope.shape[0], attn.num_local_heads, attn.qk_rope_head_dim
            )
            _, k_rope = attn.rotary_emb(positions, dummy_q, k_rope)
            pool.set_mla_kv_buffer(attn.attn_mqa, cache_loc, k_nope, k_rope)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        hidden_states = (
            self.forward_embed(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        allocator = BumpAllocator(
            buffer_size=max(1, len(self.layers) * 2),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=allocator,
            )
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.norm
            )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def compute_base_logits(self, hidden: torch.Tensor):
        if self.lm_head is None:
            raise ValueError(
                "Ling DSpark draft requires the target lm_head to be attached."
            )
        weight = self.lm_head.weight
        if self._use_fp32_lm_head and weight.is_floating_point():
            local_logits = F.linear(hidden.float(), weight.float())
        else:
            local_logits = project_through_lm_head(hidden, self.lm_head)
        return gather_and_crop_vocab(local_logits, self.lm_head), None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded = set()
        unexpected = []
        fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        cached_a_proj = {} if fuse_qkv_a_proj else None
        stacked_mapping = (
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        )
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=int(self.config.num_experts),
        )
        for checkpoint_name, loaded_weight in weights:
            if getattr(self, "is_bundled", False):
                if not checkpoint_name.startswith("dspark."):
                    continue
                checkpoint_name = checkpoint_name.removeprefix("dspark.")
            name = (
                "norm.weight"
                if checkpoint_name == "final_layernorm.weight"
                else checkpoint_name
            )
            if name.startswith("confidence_head.") and self.confidence_head is None:
                continue
            if name.startswith("layers."):
                name = name.replace(".attention.dense.", self._attention_output_alias)
                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    # The checkpoint stores the MLA down-projections separately, but
                    # q_lora_rank builds a single fused_qkv_a_proj_with_mqa.
                    cached_a_proj[name] = loaded_weight
                    q_a_name = (
                        name
                        if "q_a_proj" in name
                        else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                    )
                    kv_a_name = (
                        name
                        if "kv_a_proj_with_mqa" in name
                        else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                    )
                    if q_a_name in cached_a_proj and kv_a_name in cached_a_proj:
                        fused_name = q_a_name.replace(
                            "q_a_proj", "fused_qkv_a_proj_with_mqa"
                        )
                        param = params.get(fused_name)
                        if param is None:
                            unexpected.append(checkpoint_name)
                        else:
                            quant_config = getattr(self, "quant_config", None)
                            cat_dim = (
                                1
                                if quant_config is not None
                                and quant_config.get_name()
                                in ("awq", "awq_marlin", "moe_wna16")
                                else 0
                            )
                            fused_weight = torch.cat(
                                [cached_a_proj[q_a_name], cached_a_proj[kv_a_name]],
                                dim=cat_dim,
                            )
                            getattr(param, "weight_loader", default_weight_loader)(
                                param, fused_weight
                            )
                            loaded.add(fused_name)
                        del cached_a_proj[q_a_name]
                        del cached_a_proj[kv_a_name]
                    continue
                for param_suffix, ckpt_suffix, shard_id in stacked_mapping:
                    if ckpt_suffix not in name or ".experts." in name:
                        continue
                    candidate = name.replace(ckpt_suffix, param_suffix)
                    if candidate not in params:
                        unexpected.append(checkpoint_name)
                    else:
                        params[candidate].weight_loader(
                            params[candidate], loaded_weight, shard_id
                        )
                        loaded.add(candidate)
                    break
                else:
                    for (
                        param_suffix,
                        ckpt_suffix,
                        expert_id,
                        shard_id,
                    ) in expert_mapping:
                        if ckpt_suffix not in name:
                            continue
                        candidate = name.replace(ckpt_suffix, param_suffix)
                        if candidate not in params:
                            unexpected.append(checkpoint_name)
                        else:
                            params[candidate].weight_loader(
                                params[candidate],
                                loaded_weight,
                                candidate,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                            loaded.add(candidate)
                        break
                    else:
                        if name not in params:
                            unexpected.append(checkpoint_name)
                        else:
                            param = params[name]
                            getattr(param, "weight_loader", default_weight_loader)(
                                param, loaded_weight
                            )
                            loaded.add(name)
                continue
            if name not in params:
                unexpected.append(checkpoint_name)
                continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
            loaded.add(name)

        # Optional FP8 KV-cache scales (attn_mha/attn_mqa k_/v_scale):
        # registered as -1.0 sentinels, defaulted to 1.0 when absent, so
        # legitimately missing from the draft checkpoint.
        optional_kv_scale = (
            ".attn_mha.k_scale",
            ".attn_mha.v_scale",
            ".attn_mqa.k_scale",
            ".attn_mqa.v_scale",
        )
        quant_config = getattr(self, "quant_config", None)
        online_block_fp8 = bool(getattr(quant_config, "is_online_block_quant", False))
        missing = sorted(
            p
            for p in (set(params) - loaded)
            if not p.endswith(optional_kv_scale)
            # Online blockwise FP8 starts from BF16 weights. Its scale
            # parameters are populated by post-load quantization and therefore
            # are intentionally absent from the checkpoint.
            and not (online_block_fp8 and p.endswith("weight_scale_inv"))
        )
        if unexpected or missing:
            raise ValueError(
                "Ling DSpark checkpoint does not match the model: "
                f"unexpected={unexpected[:20]}, missing={missing[:20]}."
            )
        prepare_bailing_mla_weights(self.layers, self.quant_config)
        return loaded


EntryClass = [LingV3ForCausalLMDSparkMLA]
