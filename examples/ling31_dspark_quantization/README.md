# Ling DSpark routed-expert INT4 export

Export an unquantized, namespaced `LingV3ForCausalLMDSparkMLA` draft bundle to Humming W4A16 storage. Only routed-expert gate/up/down matrices are quantized with RTN, unsigned 4-bit storage and group size 64. Attention, shared experts, router, norms, target projection and auxiliary heads retain BF16 weights. No calibration dataset is used.

```bash
python examples/ling31_dspark_quantization/quantize_draft.py \
  --source /path/to/bf16-draft \
  --output /path/to/new-int4-draft \
  --device cuda:0 \
  --shard-mib 512 \
  --staging-dir /path/to/local-staging
```

The source is never modified and an existing output directory is rejected. Safetensors shards are staged locally before copying to the destination. `config.json` is published only after every weight shard and the weight index have been written. A directory without this final config is incomplete and must not be served.

The exported top-level and nested draft quantization configurations both specify Humming `uint4` weights, BF16 activations and `input_scale_group_size=0`.

## Serving with an MXFP4 target

Keep the target's existing launch configuration, replace its draft path with the exported directory, and select:

```bash
--speculative-algorithm DSPARK \
--speculative-draft-model-path /path/to/new-int4-draft \
--speculative-draft-model-quantization humming \
--speculative-dspark-block-size 5
```

If the target uses a global FP8 activation override, exclude the native draft prefixes (`layers.*`) from that override. Target layers use `model.layers.*` and keep their FP8 configuration:

```bash
export SGLANG_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","group_size":32,"ignored_layers":["re:^layers\\."]}'
```

This requires the regex-exclusion support in `humming_utils.py` as well as the native Ling DSpark adaptation. It does not change the target checkpoint or its weight quantization.

## Validation

Validated with Humming 0.1.18, PyTorch 2.14.1/CUDA 13, and an 8-GPU TP8/EP8 target. Exporting the tested three-layer, 512-expert draft quantized 4,608 matrices into 26 shards, reducing tensor storage from approximately 49.11 GB to 13.62 GB. The original BF16 bundle was retained.

GPU small-bundle export, exclusion-policy tests, full model loading, CUDA Graph warmup, and 16 real chat requests at concurrency 8 with 256 generated tokens each passed. These are functionality checks; no full acceptance-length, throughput or answer-quality benchmark has been performed for the INT4 draft.
