import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.model_runner_components import (
    attention_backend_setup,
)
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    ResolvedAttentionBackendStr,
    _resolve_attention_backend_strs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        # Real backends always carry this (AttentionBackend class attribute).
        self.needs_cpu_seq_lens = True


def test_split_full_attention_applies_model_wrapper_once():
    runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
        model_config=SimpleNamespace(context_len=2048),
        kv_cache_dtype=None,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        init_new_workspace=None,
    )
    wrapper_inputs = []
    wrapped_backend = object()

    def wrap_once(model_runner, backend):
        assert model_runner is runner
        wrapper_inputs.append(backend)
        return wrapped_backend

    constructors = {
        "decode-test": lambda model_runner: _FakeBackend("decode"),
        "prefill-test": lambda model_runner: _FakeBackend("prefill"),
    }
    resolved = ResolvedAttentionBackendStr(decode="decode-test", prefill="prefill-test")

    with (
        patch.dict(attention_backend_setup.ATTENTION_BACKENDS, constructors),
        patch.object(
            attention_backend_setup,
            "attn_backend_wrapper",
            side_effect=wrap_once,
        ),
    ):
        result = attention_backend_setup._build_resolved_backend(
            model_runner=runner,
            resolved=resolved,
            init_new_workspace=True,
        )

    assert result is wrapped_backend
    assert len(wrapper_inputs) == 1
    split_backend = wrapper_inputs[0]
    assert isinstance(split_backend, HybridAttnBackend)
    assert split_backend.decode_backend.name == "decode"
    assert split_backend.prefill_backend.name == "prefill"
    assert runner.init_new_workspace is True


@pytest.mark.parametrize(
    ("draft_override", "prefill", "decode", "expected_prefill", "expected_decode"),
    [
        ("dsv4", "dsv4", "dsv4", "triton", "triton"),
        (None, "dsv4", "fa3", "triton", "fa3"),
    ],
)
def test_non_dsv4_draft_falls_back_per_backend(
    draft_override, prefill, decode, expected_prefill, expected_decode
):
    server_args = SimpleNamespace(
        speculative_draft_attention_backend=draft_override,
        get_attention_backends=lambda: (prefill, decode),
        _get_default_attn_backend=lambda **kwargs: "triton",
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
    )

    resolved = _resolve_attention_backend_strs(
        server_args=server_args,
        is_draft_worker=True,
        model_config=model_config,
    )

    assert resolved.prefill == expected_prefill
    assert resolved.decode == expected_decode
    assert resolved.is_draft_override is (draft_override is not None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
