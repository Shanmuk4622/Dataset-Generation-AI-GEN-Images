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
pretty_name: AI-Generated Image Detection Dataset v3
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - real/*.parquet
    - data/*/*.parquet
- config_name: real
  data_files:
  - split: train
    path: real/*.parquet
- config_name: sd15
  data_files:
  - split: train
    path: data/sd15/*.parquet
- config_name: sdxl
  data_files:
  - split: train
    path: data/sdxl/*.parquet
- config_name: flux_schnell
  data_files:
  - split: train
    path: data/flux_schnell/*.parquet
- config_name: kandinsky22
  data_files:
  - split: train
    path: data/kandinsky22/*.parquet
- config_name: pixart_sigma
  data_files:
  - split: train
    path: data/pixart_sigma/*.parquet
- config_name: wuerstchen
  data_files:
  - split: train
    path: data/wuerstchen/*.parquet
---

# AI-Generated Image Detection Dataset (v3)

**Paired real / AI images for training and evaluating AI-generated-image detectors.**

Every real image is matched with **one AI-generated partner per generator**, and each
pair shares a single, image-grounded caption. Because the real image and all of its
synthetic partners depict the *same scene* described by the *same prompt*, a detector
trained here is pushed toward the **synthesis fingerprint** (texture, frequency and
rendering artefacts) instead of scene content — the shortcut that inflates scores on
naively-built datasets.

> **v3 note:** v3 is identical in content to v2, but the `image` column is re-encoded
> into the Hugging Face `Image` type so thumbnails render correctly in the Dataset
> Viewer / Data Studio. No pixels were changed — the underlying PNG bytes are byte-for-byte
> the same as v2.

## At a glance

| | |
|---|---|
| Real images | 10,000 (COCO + ImageNet-1k), label `0` |
| AI images | 60,000 across 6 generators, label `1` |
| Pairs | 10,000 (each real has 6 AI partners) |
| Resolution | 512×512 RGB PNG |
| Preprocessing | Identical canonical pipeline for real **and** AI (`pipeline_version=1.2`) |
| Master seed | `42` — controls generation seeds and split assignment |
| Split | 70 / 15 / 15 pair-level train / val / test |
| License | Non-commercial research use (most-restrictive inherited term) |

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

Real images were captioned with **Salesforce/blip2-opt-2.7b**, cleaned and capped to
75 CLIP tokens, then reused **byte-identically** by all six generators. The column
`source_real_id` links every AI image back to its real partner
(`source_real_id == image_id` for real rows).

## Canonical preprocess (leak-free)

Both real and AI images pass through the **same** pipeline:

1. EXIF-transpose → convert to RGB → center-crop to square → Lanczos resize to 512.
2. JPEG-equalise (quality 95, 4:4:4) → save as PNG (compress level 6, no EXIF/ICC).

Applying an identical transform to both classes removes the resolution, colour-space
and JPEG-artefact shortcuts that would otherwise let a model "cheat" instead of learning
the generative fingerprint.

## Splits

Deterministic **pair-level** split keyed on `source_real_id` (seed `42`): a real image
and all six of its AI partners always land in the same fold, so there is **no scene
leakage** between train / val / test.

| Split | Images |
|-------|--------|
| train | 49,392 |
| val   | 10,122 |
| test  | 10,486 |

The row-level split label lives in `manifest.parquet` (and `splits.parquet` keyed by
`source_real_id`).

## Dataset structure

```
ai-detection-dataset-v3/
├── real/                       # 10k real images, label 0  (real-*.parquet)
├── data/
│   ├── sd15/                   # 10k AI images per generator, label 1
│   ├── sdxl/
│   ├── flux_schnell/
│   ├── kandinsky22/
│   ├── pixart_sigma/
│   └── wuerstchen/
├── manifest.parquet            # full index: id, pairing key, label, generator, split (no bytes)
├── splits.parquet              # source_real_id -> split
├── config.json                 # build configuration (seeds, models, params)
├── validation_report.json      # leak-audit / sniff-test results
└── ai_detect_common.py         # shared helper library used by the build notebooks
```

The Dataset Viewer exposes one **`default`** config (real + all generators combined) plus
one config **per generator** and a **`real`** config, so you can browse each source on its
own.

## Usage

### With `datasets` (streaming, viewer-native)

```python
from datasets import load_dataset

# combined real + AI
ds = load_dataset("Shanmuk4622/ai-detection-dataset-v3", split="train", streaming=True)
ex = next(iter(ds))
ex["image"]      # PIL.Image (decoded automatically)
ex["label"]      # 0 = real, 1 = AI

# just one generator
flux = load_dataset("Shanmuk4622/ai-detection-dataset-v3", "flux_schnell", split="train")
```

### With `pyarrow` (raw bytes, no decode)

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image
import io

shard = hf_hub_download("Shanmuk4622/ai-detection-dataset-v3",
                        "data/flux_schnell/flux_schnell-301fd92e-00002.parquet",
                        repo_type="dataset")
tbl = pq.read_table(shard)
img = Image.open(io.BytesIO(tbl["image"][0]["bytes"].as_py()))   # v3: image is a struct
```

> **Migrating from v2?** In v2 the `image` column was raw `binary`
> (`tbl["image"][0].as_py()`). In v3 it is a `struct{bytes, path}`, so read
> `tbl["image"][0]["bytes"].as_py()`.

## Schema

| column | type | description |
|--------|------|-------------|
| `image` | Image `struct{bytes, path}` | canonical 512×512 RGB PNG |
| `image_id` | string | globally unique, e.g. `real_000001` / `sdxl_000001` |
| `source_real_id` | string | pairing key; equals `image_id` for real rows |
| `label` | int8 | 0 = real, 1 = AI |
| `generator` | string | `real`, `sd15`, `sdxl`, `flux_schnell`, `kandinsky22`, `pixart_sigma`, `wuerstchen` |
| `source_dataset` | string | `coco`, `imagenet`, or `<generator>` |
| `prompt` | string | caption used (AI rows); null for real |
| `width` / `height` | int16 | always 512 |
| `orig_width` / `orig_height` | int32 | provenance only — **never use as a training feature** |
| `gen_model_id` | string | HF model id |
| `gen_steps` / `gen_guidance` / `gen_native_res` | int16 / float32 / int16 | generation params |
| `seed` | int64 | `hash(master_seed, generator, source_real_id)` |
| `caption_model` | string | captioner used |
| `pipeline_version` | string | `1.2` |
| `sha256` | string | hex digest of the PNG bytes |
| `created_utc` | string | ISO-8601 UTC timestamp |

## Leak audit

A "sniff test" trains a lightweight classifier on each generator vs. real using only
low-level statistics and checks that separability stays in a healthy band
(`<0.85` healthy, `0.85–0.95` investigate, `>0.95` leak). Latest results:

| generator | score |
|-----------|-------|
| sd15 | 0.910 |
| sdxl | 0.973 |
| flux_schnell | 0.917 |
| kandinsky22 | 0.958 |
| pixart_sigma | 0.943 |
| wuerstchen | 0.843 |
| **overall** | **0.860** |

Full results in `validation_report.json`.

## Provenance, license and intended use

- **ImageNet** content is **non-commercial research only** (ILSVRC terms).
- **COCO** images are Flickr-sourced (Creative Commons).
- AI images are synthetic outputs of the models listed above, each under its own license.
- This dataset inherits the **most restrictive** applicable term: **non-commercial
  research use**. *Not legal advice.*
- Intended for training / evaluating AI-image detectors.

## Limitations and caveats

- The core set is **text-to-image only** (no img2img / reference conditioning); detectors
  trained here target the text-to-image threat model.
- Caption quality (BLIP-2 concise sentences) bounds prompt diversity.
- FLUX.1-schnell and Würstchen run with CPU offload on T4; generation conditions are
  otherwise consistent with standard inference settings.
- Per-model step counts were reduced for speed (SDXL 8 steps, SD 1.5 20 steps), which is
  representative of real-world fast inference.

## Citation

```bibtex
@misc{aidetection_dataset_v3,
  title  = {AI-Generated Image Detection Dataset (v3)},
  author = {Shanmuk4622},
  year   = {2025},
  howpublished = {\url{https://huggingface.co/datasets/Shanmuk4622/ai-detection-dataset-v3}}
}
```
