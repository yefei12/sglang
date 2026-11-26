#!/usr/bin/env python3
"""Export a namespaced Ling 3.1 DSpark BF16 draft to Humming weight storage."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

INT4_CONFIG = {
    "quant_method": "humming",
    "group_size": 64,
    "ignored_layers": ["attention", "shared_experts"],
    "b_dtype": "uint4",
    "a_dtype": "bfloat16",
    "input_scale_group_size": 0,
}


def should_quantize(name, tensor):
    """Quantize only the namespaced draft's routed expert projections."""
    return (
        tensor.ndim == 2
        and name.startswith("dspark.layers.")
        and ".mlp.experts." in name
        and any(
            name.endswith(f".{projection}.weight")
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-mib", type=int, default=512)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="Local filesystem used by safetensors before copying to shared storage",
    )
    args = parser.parse_args()
    if args.shard_mib <= 0:
        parser.error("--shard-mib must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    import torch
    from humming.schema.humming import HummingWeightSchema
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((args.source / "config.json").read_text())
    if config.get("quantization_config") or config.get("dspark_config", {}).get(
        "quantization_config"
    ):
        parser.error("Expected the original unquantized BF16 draft")
    if (
        config.get("dspark_bundle_format") != "namespaced_safetensors"
        or config.get("dspark_bundle_version") != 1
        or config.get("dspark_config", {}).get("architectures")
        != ["LingV3ForCausalLMDSparkMLA"]
    ):
        parser.error(
            "Only the tested namespaced Ling DSpark bundle format is supported"
        )
    source_weights = args.source / "draft_model.safetensors"
    if not source_weights.is_file():
        parser.error("Missing draft_model.safetensors")
    quant_config = INT4_CONFIG.copy()
    schema = HummingWeightSchema(
        b_dtype=quant_config["b_dtype"],
        weight_scale_group_size=quant_config["group_size"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    for item in args.source.iterdir():
        if item.is_file() and (
            item.suffix == ".py"
            or item.name.startswith("tokenizer")
            or item.name
            in (
                "special_tokens_map.json",
                "generation_config.json",
                "chat_template.jinja",
            )
        ):
            shutil.copy2(item, args.output / item.name)
    pending, weight_map = {}, {}
    pending_bytes = total_bytes = shard_id = quantized = 0
    with tempfile.TemporaryDirectory(dir=args.staging_dir) as staging:

        def flush():
            nonlocal pending, pending_bytes, shard_id, total_bytes
            if not pending:
                return
            shard_id += 1
            name = f"model-{shard_id:05d}.safetensors"
            local_file = Path(staging) / name
            save_file(pending, str(local_file))
            # Direct safetensors writes are unsupported on some /mnt mounts.
            shutil.copyfile(local_file, args.output / name)
            local_file.unlink()
            for key, value in pending.items():
                weight_map[key] = name
                total_bytes += value.numel() * value.element_size()
            pending, pending_bytes = {}, 0

        with safe_open(str(source_weights), framework="pt", device="cpu") as source:
            keys = list(source.keys())
            for index, name in enumerate(keys):
                tensor = source.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        f"Expected BF16 source tensor: {name}: {tensor.dtype}"
                    )
                if should_quantize(name, tensor):
                    with torch.no_grad():
                        converted = schema.quant_tensor(
                            tensor.to(args.device),
                            schema,
                            torch.bfloat16,
                        )
                    tensors = {
                        name[: -len("weight")] + suffix: value.cpu().contiguous()
                        for suffix, value in converted.items()
                    }
                    quantized += 1
                else:
                    tensors = {name: tensor}
                for key, value in tensors.items():
                    pending[key] = value
                    pending_bytes += value.numel() * value.element_size()
                if pending_bytes >= args.shard_mib * 1024**2:
                    flush()
                if index % 100 == 0:
                    print(
                        f"{index + 1}/{len(keys)} tensors; quantized={quantized}",
                        flush=True,
                    )
        flush()
    if not quantized:
        raise ValueError("Source contains no routed expert projections to quantize")
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": weight_map,
            },
            indent=2,
        )
    )
    config["quantization_config"] = quant_config.copy()
    config["dspark_config"]["quantization_config"] = quant_config.copy()
    # Publish the usable model config after all weight shards have been written.
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    manifest = {
        "source": str(args.source.resolve()),
        "mode": "moe-int4",
        "method": "Humming RTN W4A16 routed-expert export",
        "source_weight_bytes": source_weights.stat().st_size,
        "quantized_matrices": quantized,
        "tensor_bytes": total_bytes,
        "configuration": quant_config,
        "preserved": "all attention, shared experts, router, norms, target projection, and auxiliary heads",
    }
    (args.output / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
