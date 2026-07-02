---
license: other
task_categories:
- image-classification
language:
- en
tags:
- ai-generated-image-detection
- synthetic-image-detection
- diffusion-models
- image-forensics
pretty_name: AI-Image Detection Dataset
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - real/train/*.parquet
    - sd15/train/*.parquet
    - sdxl/train/*.parquet
    - flux_schnell/train/*.parquet
    - kandinsky22/train/*.parquet
    - pixart_sigma/train/*.parquet
    - wuerstchen/train/*.parquet
  - split: validation
    path:
    - real/val/*.parquet
    - sd15/val/*.parquet
    - sdxl/val/*.parquet
    - flux_schnell/val/*.parquet
    - kandinsky22/val/*.parquet
    - pixart_sigma/val/*.parquet
    - wuerstchen/val/*.parquet
  - split: test
    path:
    - real/test/*.parquet
    - sd15/test/*.parquet
    - sdxl/test/*.parquet
    - flux_schnell/test/*.parquet
    - kandinsky22/test/*.parquet
    - pixart_sigma/test/*.parquet
    - wuerstchen/test/*.parquet
- config_name: real
  data_files:
  - split: train
    path: real/train/*.parquet
  - split: validation
    path: real/val/*.parquet
  - split: test
    path: real/test/*.parquet
- config_name: sd15
  data_files:
  - split: train
    path: sd15/train/*.parquet
  - split: validation
    path: sd15/val/*.parquet
  - split: test
    path: sd15/test/*.parquet
- config_name: sdxl
  data_files:
  - split: train
    path: sdxl/train/*.parquet
  - split: validation
    path: sdxl/val/*.parquet
  - split: test
    path: sdxl/test/*.parquet
- config_name: flux_schnell
  data_files:
  - split: train
    path: flux_schnell/train/*.parquet
  - split: validation
    path: flux_schnell/val/*.parquet
  - split: test
    path: flux_schnell/test/*.parquet
- config_name: kandinsky22
  data_files:
  - split: train
    path: kandinsky22/train/*.parquet
  - split: validation
    path: kandinsky22/val/*.parquet
  - split: test
    path: kandinsky22/test/*.parquet
- config_name: pixart_sigma
  data_files:
  - split: train
    path: pixart_sigma/train/*.parquet
  - split: validation
    path: pixart_sigma/val/*.parquet
  - split: test
    path: pixart_sigma/test/*.parquet
- config_name: wuerstchen
  data_files:
  - split: train
    path: wuerstchen/train/*.parquet
  - split: validation
    path: wuerstchen/val/*.parquet
  - split: test
    path: wuerstchen/test/*.parquet
---

# AI-Image Detection Dataset

**Paired real / AI images, with shared image-grounded captions, for training and
evaluating AI-generated-image detectors.**

Each of 10,000 real photos is captioned once (BLIP-2) and paired with **one synthetic
partner per generator** (6 generators → 60,000 AI images). A real image and all of its
AI partners **share the same prompt**, so the *only* systematic difference between the
classes is the generative process itself. A detector trained here is pushed toward the
**synthesis fingerprint** (texture, frequency, rendering artefacts) instead of scene
content — the shortcut that inflates accuracy on naively-built datasets.

> This is the clean, self-contained release. Every row is complete on its own: the
> **image renders in the viewer**, carries its **prompt**, its **train/val/test split**,
> and full **caption + generation provenance** — no need to join against side files. The
> viewer exposes a **`default` subset (all 70k)** plus **one subset per model**, each with
> its own train/validation/test splits.

## At a glance

| | |
|---|---|
| Real images | 10,000 (COCO + ImageNet-1k), label `0` |
| AI images | 60,000 across 6 generators, label `1` |
| Total rows | 70,000 |
| Pairs | 10,000 (each real ↔ 6 AI partners, shared caption) |
| Resolution | 512×512 RGB PNG |
| Splits | train 49,392 / validation 10,122 / test 10,486 (pair-level, seed 42) |
| Subsets | `default` + `real`, `sd15`, `sdxl`, `flux_schnell`, `kandinsky22`, `pixart_sigma`, `wuerstchen` |
| Preprocessing | Identical canonical pipeline for real **and** AI (`pipeline_version 1.2`) |
| License | Non-commercial research use (most-restrictive inherited term) |

## Subsets

The Dataset Viewer and `load_dataset` expose eight subsets, each split into
train / validation / test:

| subset | rows | contents |
|--------|------|----------|
| `default` | 70,000 | everything (real + all generators) |
| `real` | 10,000 | real photos only (label 0) |
| `sd15` | 10,000 | Stable Diffusion 1.5 outputs |
| `sdxl` | 10,000 | SDXL outputs |
| `flux_schnell` | 10,000 | FLUX.1-schnell outputs |
| `kandinsky22` | 10,000 | Kandinsky 2.2 outputs |
| `pixart_sigma` | 10,000 | PixArt-Σ outputs |
| `wuerstchen` | 10,000 | Würstchen outputs |

## What changed vs. the earlier release

- **`image` is now the Hugging Face `Image` type** → thumbnails render in the Dataset Viewer.
- **`prompt` is populated on real rows** (each real image's own BLIP-2 caption), so a pair
  literally shares one caption.
- **`split` is a column** and the files are organised into `<subset>/{train,val,test}/`, so
  `load_dataset(..., split=...)` and the viewer's split tabs work.
- **Per-model subsets** are selectable directly in the viewer / loader.
- **Caption provenance added**: `raw_caption`, `caption_n_tokens`, `caption_model`, plus the
  original BLIP-2 caption shards under `captions/`.

## Generators

| key | model id | native res | steps | guidance |
|-----|----------|-----------|-------|----------|
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512 | 20 | 7.0 |
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | 8 | 7.0 |
| `flux_schnell` | `black-forest-labs/FLUX.1-schnell` | 1024 | 4 | 0.0 |
| `kandinsky22` | `kandinsky-community/kandinsky-2-2-decoder` | 768 | 25 | 4.0 |
| `pixart_sigma` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | 1024 | 20 | 4.5 |
| `wuerstchen` | `warp-ai/wuerstchen` | 1024 | [20, 10] | [4.0, 0.0] |

## How the pairs are built

Real images are captioned with **Salesforce/blip2-opt-2.7b**, the caption is cleaned and
capped to 75 CLIP tokens, and that **exact caption is reused by all six generators** and
also stored on the real row. `source_real_id` links every AI image back to its real
partner (`source_real_id == image_id` for real rows).

## Canonical preprocess (leak-free)

Both real and AI images pass through the **same** pipeline:

1. EXIF-transpose → convert to RGB → center-crop to square → Lanczos resize to 512.
2. JPEG-equalise (quality 95, 4:4:4) → save as PNG (compress level 6, no EXIF/ICC).

Applying an identical transform to both classes removes the resolution, colour-space and
JPEG-artefact shortcuts that would otherwise let a model cheat.

## Splits

Deterministic **pair-level** split keyed on `source_real_id` (seed `42`): a real image and
all six of its AI partners always land in the same fold, so there is **no scene leakage**
between train / validation / test.

| split | images |
|-------|--------|
| train | 49,392 |
| validation | 10,122 |
| test | 10,486 |

## Dataset structure

```
ai-image-detection-dataset/
├── real/            {train,val,test}/*.parquet    # 10k real photos, label 0
├── sd15/            {train,val,test}/*.parquet    # 10k AI per generator, label 1
├── sdxl/            {train,val,test}/*.parquet
├── flux_schnell/    {train,val,test}/*.parquet
├── kandinsky22/     {train,val,test}/*.parquet
├── pixart_sigma/    {train,val,test}/*.parquet
├── wuerstchen/      {train,val,test}/*.parquet
├── captions/        captions-*.parquet            # original BLIP-2 caption shards (provenance)
├── metadata/
│   ├── manifest.parquet        # id, pairing key, label, generator, split (no image bytes)
│   ├── splits.parquet          # source_real_id -> split
│   ├── captions.parquet        # consolidated real-image captions + provenance
│   ├── config.json             # build configuration (seeds, models, params)
│   └── validation_report.json  # leak-audit / sniff-test results
├── ai_detect_common.py         # shared preprocess + schema library
└── README.md
```

## Usage

```python
from datasets import load_dataset

# everything, by split
ds = load_dataset("Shanmuk4622/ai-image-detection-dataset")          # default subset
train, val, test = ds["train"], ds["validation"], ds["test"]

ex = train[0]
ex["image"]     # PIL.Image (decoded automatically, renders in the viewer)
ex["label"]     # 0 = real, 1 = AI
ex["prompt"]    # shared caption (now present on real rows too)
ex["generator"] # 'real' or one of the 6 generators
ex["split"]     # 'train' | 'validation' | 'test'

# a single model's subset (real + one generator are separate subsets)
flux_test = load_dataset("Shanmuk4622/ai-image-detection-dataset", "flux_schnell", split="test")
real_train = load_dataset("Shanmuk4622/ai-image-detection-dataset", "real", split="train")
```

## Schema

| column | type | description |
|--------|------|-------------|
| `image` | Image `struct{bytes, path}` | canonical 512×512 RGB PNG |
| `image_id` | string | globally unique, e.g. `real_000001` / `sdxl_000001` |
| `source_real_id` | string | pairing key; equals `image_id` for real rows |
| `label` | int8 | 0 = real, 1 = AI |
| `generator` | string | `real`, `sd15`, `sdxl`, `flux_schnell`, `kandinsky22`, `pixart_sigma`, `wuerstchen` |
| `source_dataset` | string | `coco`, `imagenet`, or `<generator>` |
| `split` | string | `train` / `validation` / `test` |
| `prompt` | string | shared image-grounded caption (**now populated for real rows**) |
| `raw_caption` | string | uncleaned BLIP-2 output |
| `caption_n_tokens` | int16 | CLIP-token length of `prompt` |
| `caption_model` | string | captioner used (`Salesforce/blip2-opt-2.7b`) |
| `width` / `height` | int16 | always 512 |
| `orig_width` / `orig_height` | int32 | provenance only — **never use as a training feature** |
| `gen_model_id` | string | HF model id (null for real) |
| `gen_steps` / `gen_guidance` / `gen_native_res` | int16 / float32 / int16 | generation params (null for real) |
| `seed` | int64 | `hash(master_seed, generator, source_real_id)` (null for real) |
| `pipeline_version` | string | `1.2` |
| `sha256` | string | hex digest of the PNG bytes |
| `created_utc` | string | ISO-8601 UTC timestamp |

## Leak audit

A "sniff test" trains a lightweight classifier on each generator vs. real using only
low-level statistics and checks separability stays healthy (`<0.85` healthy,
`0.85–0.95` investigate, `>0.95` leak):

| generator | score |
|-----------|-------|
| sd15 | 0.910 |
| sdxl | 0.973 |
| flux_schnell | 0.917 |
| kandinsky22 | 0.958 |
| pixart_sigma | 0.943 |
| wuerstchen | 0.843 |
| **overall** | **0.860** |

Full results in `metadata/validation_report.json`.

## Provenance, license and intended use

- **ImageNet** content is **non-commercial research only** (ILSVRC terms).
- **COCO** images are Flickr-sourced (Creative Commons).
- AI images are synthetic outputs of the models listed above, each under its own license.
- This dataset inherits the **most restrictive** applicable term: **non-commercial
  research use**. *Not legal advice.*

## Limitations and caveats

- Core set is **text-to-image only** (no img2img / reference conditioning); detectors
  trained here target the text-to-image threat model.
- Caption quality (BLIP-2 concise sentences) bounds prompt diversity.
- FLUX.1-schnell and Würstchen run with CPU offload on T4; generation conditions are
  otherwise consistent with standard inference settings.
- Per-model step counts were reduced for speed (SDXL 8 steps, SD 1.5 20 steps), which is
  representative of real-world fast inference.

## Citation

```bibtex
@misc{aiimage_detection_dataset,
  title  = {AI-Image Detection Dataset},
  author = {Shanmuk4622},
  year   = {2025},
  howpublished = {\url{https://huggingface.co/datasets/Shanmuk4622/ai-image-detection-dataset}}
}
```
