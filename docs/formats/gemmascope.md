# GemmaScope — Format Reference

Concise reference for loading **GemmaScope** transcoders (JumpReLU SAEs for Gemma 2) via anamnesis. Distilled from hands-on investigation on 2026-04-18 and cross-checked against Anthropic's `circuit-tracer` reference loader.

## Contents

- [The two-repo gotcha](#the-two-repo-gotcha)
- [NPZ tensor layout](#npz-tensor-layout)
- [Picking a variant per layer](#picking-a-variant-per-layer)
- [Loading via anamnesis](#loading-via-anamnesis)
- [Reference implementations](#reference-implementations)
- [Where to find files on HuggingFace](#where-to-find-files-on-huggingface)
- [Canonical sources](#canonical-sources)
- [The `.bin` format (for completeness)](#the-bin-format-for-completeness)

---

## The two-repo gotcha

GemmaScope ships across **two** HuggingFace repos. Loading the wrong one gives you the wrong thing:

| Repo | Contents | Use it for |
|---|---|---|
| `google/gemma-scope-2b-pt-transcoders` | 26 NPZ files, one per layer × L0 variant, ~288 MiB each, FP32 | **Loading the transcoder to run it.** |
| `mntss/gemma-scope-transcoders` | `config.yaml` + 26 `features/layer_{l}.bin` (gzipped-JSON chunks) | **Feature dashboards only.** Per-feature activation quantiles, top/bottom logits, `act_min`/`act_max`. **Not weights.** |

**If you want to run a transcoder, skip `mntss/gemma-scope-transcoders` entirely.** Its `config.yaml` is only useful as an index that maps each layer to the recommended (lowest-L0) NPZ variant in the Google repo. Its `.bin` files are Neuronpedia-style feature explanations, not tensors.

This distinction is undocumented in both upstream READMEs. `circuit-tracer`'s loader silently ignores the `mntss` repo and downloads directly from Google. Anyone starting from `mntss` first (e.g. via `hf-fm list-files`) hits a wall without this signpost.

---

## NPZ tensor layout

Each `params.npz` file in `google/gemma-scope-2b-pt-transcoders/layer_N/width_16k/average_l0_X/` contains:

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `W_enc` | `[d_model, n_features]` | F32 | Encoder. **Transposed** vs Llama/Qwen PLT convention (which uses `[n_features, d_model]`). |
| `W_dec` | `[n_features, d_model]` | F32 | Decoder. Rank-2 (per-layer PLT, not cross-layer). |
| `b_enc` | `[n_features]` | F32 | Encoder bias. |
| `b_dec` | `[d_model]` | F32 | Decoder bias. |
| `threshold` | `[n_features]` | F32 | JumpReLU gate. Inference activation is `pre * (pre > threshold)` element-wise, where `pre = W_enc·x + b_enc`. (Circuit-tracer constructs `JumpReLU(threshold, 0.1)` — the `0.1` is the straight-through-estimator bandwidth, used only during training.) |

For Gemma 2 2B the dimensions are `d_model = 2304`, `n_features = 16384` (`width_16k`).

**Not present:** `W_skip`. GemmaScope is a pure JumpReLU transcoder without the MLP skip path that `mntss/transcoder-Llama-*` and `mwhanna/qwen3-*` include.

---

## Picking a variant per layer

`google/gemma-scope-2b-pt-transcoders` ships multiple L0 variants per layer (different sparsity levels: ~10, ~50, ~200, ~800 avg active features). The canonical lowest-L0 set — what `mntss/gemma-scope-transcoders/config.yaml` curates — is:

```yaml
transcoders:
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_0/width_16k/average_l0_76/params.npz"
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_1/width_16k/average_l0_65/params.npz"
  # ... 26 entries total
  - "hf://google/gemma-scope-2b-pt-transcoders/layer_25/width_16k/average_l0_41/params.npz"
```

The L0 value varies per layer (5 → ~76 depending on which layer is easiest to sparsify). Parsing this YAML is the cheapest way to get the recommended URL set.

---

## Loading via anamnesis

GemmaScope NPZ files load through anamnesis's existing NPZ parser (Phase 3, v0.3.0). No new anamnesis feature is needed:

```rust
use anamnesis::parse::npz::parse_npz;

// parse_npz returns HashMap<String, NpzTensor>
let tensors = parse_npz("layer_11/width_16k/average_l0_5/params.npz")?;

let w_enc     = &tensors["W_enc"];      // shape [2304, 16384] F32 — transposed!
let w_dec     = &tensors["W_dec"];      // shape [16384, 2304] F32
let b_enc     = &tensors["b_enc"];      // shape [16384]       F32
let b_dec     = &tensors["b_dec"];      // shape [2304]        F32
let threshold = &tensors["threshold"];  // shape [16384]       F32 — JumpReLU gate

// Each NpzTensor exposes: name: String, shape: Vec<usize>, dtype: NpzDtype, data: Vec<u8>.
// `data` is raw row-major F32 little-endian bytes — feed directly into your tensor library.
```

**Things your downstream crate (not anamnesis) must handle:**
- Transpose `W_enc` to match whatever orientation your encoder path expects
- Apply JumpReLU with `threshold` as a per-feature gate (not plain ReLU)
- Fetch the YAML config from `mntss/gemma-scope-transcoders` (or hard-code URLs)

---

## Reference implementations

| Language | Code | Notes |
|---|---|---|
| Python | [`circuit-tracer/circuit_tracer/transcoder/single_layer_transcoder.py`](https://github.com/safety-research/circuit-tracer/blob/main/circuit_tracer/transcoder/single_layer_transcoder.py) — function `load_gemma_scope_transcoder()` | Canonical loader. Uses `hf_hub_download` + `np.load()`. Builds `JumpReLU(threshold, 0.1)`. |
| Rust | candle-mi `TranscoderSchema::GemmaScopeNpz` (v0.1.10+, see `candle-mi/docs/roadmaps/candle_mi_v019_roadmap_V3.md`) | Mirrors the circuit-tracer path using anamnesis NPZ + a YAML config parser. Handles the `W_enc` transpose and JumpReLU `threshold` application. |

Circuit-tracer uses the key name `"activation_function.threshold"` in its own state-dict convention; the NPZ on disk simply names it `threshold`. Rename at load time if you're mixing conventions.

---

## Where to find files on HuggingFace

Beyond [`google/gemma-scope-2b-pt-transcoders`](https://huggingface.co/google/gemma-scope-2b-pt-transcoders) (the focus of this doc), the wider GemmaScope family is spread across many HuggingFace repos. Run `hf-fm search gemmascope` (or `huggingface-cli search gemmascope`) to enumerate them.

Official Google releases follow the slug convention `google/gemma-scope-{size}-{tune}-{site}`:

| Slug part | Values | Meaning |
|---|---|---|
| `size` | `2b`, `9b`, `27b`, `2-270m`, `2-1b` | Backing Gemma 2 model size |
| `tune` | `pt`, `it` | Pretrained vs instruction-tuned base |
| `site` | `res`, `att`, `mlp`, `transcoders` | Hook site: residual stream, attention output, MLP output, or MLP transcoder |

So `google/gemma-scope-9b-pt-mlp` is "JumpReLU SAE on the MLP output of pretrained Gemma 2 9B". All Google releases carry the `saelens` tag.

A handful of community / third-party ports show up alongside the official set: `mwhanna/gemma-scope-transcoders`, `mwhanna/gemma-scope-attn-saes-16k`, `EleutherAI/gemmascope-transcoders-sparsify`, `weijie210/gemma-scope-2b-it-pt-res`. Verify their tensor names and dtypes before assuming they match the layout in [NPZ tensor layout](#npz-tensor-layout) — that section was checked only against the Google `2b-pt-transcoders` repo.

---

## Canonical sources

- **Paper:** [Lieberum et al., "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"](https://arxiv.org/abs/2408.05147) — Figure 12 has the TransformerLens load snippet.
- **Weights:** [`google/gemma-scope-2b-pt-transcoders`](https://huggingface.co/google/gemma-scope-2b-pt-transcoders)
- **Feature metadata (not weights):** [`mntss/gemma-scope-transcoders`](https://huggingface.co/mntss/gemma-scope-transcoders)
- **Reference loader:** [`safety-research/circuit-tracer`](https://github.com/safety-research/circuit-tracer)

---

## The `.bin` format (for completeness)

If you ever do want to read `mntss/gemma-scope-transcoders/features/layer_{l}.bin` (e.g. to build a feature-dashboard UI), the format is:

```
file     := 16384 × chunk           (one chunk per feature)
chunk    := [u32 LE: gzip_length] [gzip-compressed JSON]
JSON     := {
  "index":              int,
  "examples_quantiles": [{...}, ...10 quantile buckets],
  "top_logits":         [string, string, string, string, string],
  "bottom_logits":      [string, string, string, string, string],
  "act_min":            float,
  "act_max":            float
}
```

The `features/index.json.gz` sibling file provides the byte-offset table (`offsets[i]` → start of feature `i`'s chunk). Version identifier: `{"version": "1.0", "format": "variable_chunks"}`.

**This format is orthogonal to running the transcoder.** Reading it requires gzip + JSON only — no new parser primitives. It's not a priority for anamnesis; a companion interpretability-tooling crate is the natural home if this ever becomes a real need.
