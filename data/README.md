# Dataset and provenance

Runtime images are deliberately excluded from Git. The preparation command
streams only the selected subset into:

- `raw/authentic/` — SID_Set label `0` (real)
- `raw/generated/` — SID_Set label `1` (fully synthetic)
- `processed/manifest.csv` — ordered training/evaluation index
- `processed/manifest_summary.json` — source, seed, class/split counts, and skips

## Source and license

The training source is
[`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set), whose
dataset card lists a CC BY 4.0 license. SID_Set contains label `0` (real), label
`1` (fully synthetic), and label `2` (tampered). This binary prototype retains
only labels 0 and 1 and streams from the upstream `train` split.

Please cite the SID_Set/SIDA authors as requested on the dataset card. The
organizer-provided WildFake demonstration subset must remain isolated from
training, threshold selection, and model selection.

## Leakage and shortcut controls

Preparation is deterministic for a fixed seed and:

- pins new SID_Set preparations to Hugging Face revision
  `dc03ead57929879319ce30a82bfcfb8d317b10bd` (legacy prepared subsets retain
  an explicit unrecorded revision rather than receiving a revision after the fact);
- selects equal numbers of labels 0 and 1;
- hashes the normalized JPEG bytes and rejects duplicates in the exact training
  representation (including distinct inputs that collapse during JPEG encoding);
- keeps every shared SID `img_id` in one local split;
- creates class-balanced `train`, `val`, and `test` splits; and
- re-encodes both classes as RGB JPEG quality 95 with 4:4:4 subsampling, reducing
  class-specific container/format shortcuts.

The retained SID subset still has a strong source-geometry imbalance: the fully
synthetic samples are square while most authentic samples are not. The shared
model preprocessing therefore preserves aspect ratio with a short-edge resize
followed by a center crop; it never stretches every source directly to a square.
This is designed to reduce that shortcut, but it does not turn an internal SID
holdout into an unseen-generator benchmark. Public claims must state the
evaluated dataset and scope explicitly.

The generated manifest summary is the authoritative record of the counts from
the actual run. Do not replace it with estimated numbers in the submission.

## Optional Unbiased Tiny GenImage v2 training source

The v2 candidate workflow uses Kaggle's
[`cartografia/unbiased-tiny-genimage`](https://www.kaggle.com/datasets/cartografia/unbiased-tiny-genimage)
version 1 as a read-only attached input. The workflow records the dataset's
CC BY-NC-SA 4.0 notice and the participant's 2026-08-31 licence confirmation;
users remain responsible for complying with the upstream
[`GenImage licence`](https://github.com/GenImage-Dataset/GenImage/blob/main/License)
and [`ImageNet access agreement`](https://image-net.org/accessagreement).

Preparation fails unless the attached input has exactly 23,329 files and
2,528,629,592 bytes: 2,500 images from each of ADM, BigGAN, GLIDE, Midjourney,
Stable Diffusion 1.5, VQDM, and Wukong; 5,828 real Nature images; and the pinned
`nature_metadata.csv` SHA-256
`5f9a46e53e624339f6db8cc4d4a4fe5e54a0371e4b07a7da278300f6ed699e91`.

With seed 42, the script selects 800 generated images per generator and 5,600
source-class-balanced real images. Both labels are decoded and re-encoded as RGB
JPEG quality 95 with 4:4:4 subsampling. It removes same-label normalized-image
duplicates against the complete pinned SID sample and within GenImage, refills
the quotas from the deterministic candidate ordering, and rejects conflicting
labels, corrupt files, unsafe paths, or an unexpected inventory.

The exact GenImage split is:

| Split | Real | Generated | Total |
| --- | ---: | ---: | ---: |
| Train | 4,480 | 4,480 | 8,960 |
| Validation | 560 | 560 | 1,120 |
| Test | 560 | 560 | 1,120 |

Only the 4,800 SID training rows are added to the GenImage training rows. SID
validation/test rows remain outside v2 training, and the combined validation and
test splits are the untouched GenImage splits. Images, detailed manifests,
predictions, and audit outputs remain ignored; only compact provenance, digests,
and reviewed aggregate evidence may be committed. WildFake is neither read nor
used anywhere in this workflow.

The completed T4 run is recorded by
`processed/genimage_v2_manifest_summary.json`. Its candidate checkpoint was not
promoted: held-out GenImage generalization improved substantially, but SID
false positives increased at the fixed 0.50 threshold. Aggregate evidence is
published under `artifacts/metrics/genimage_v2_summary.json`; per-image
predictions remain in the ignored local audit export.

## WildFake post-lock demonstration subset

WildFake is an external, demonstration-only benchmark. It is never an input to
training, checkpoint selection, early stopping, calibration, or threshold
selection. The model checkpoint and `0.50` decision threshold were frozen before
this data was downloaded.

The exact subset is pinned to ModelScope revision
`18f53ff36ad9da60644039f0452b0e7b3907af6f` and contains:

- 4,998 authentic images selected by `/val2017/` from `real_coco.csv`;
- 8,843 generated images selected with `IsAdvanced=1` and `IsFake=1` from
  `dalle3.csv`.

The downloader verifies the metadata SHA-256 values, remotely inspects the ZIP
central directories, and downloads only one contiguous byte range from each
large archive. Every extracted file must pass local-header, compression, size,
CRC-32, image-decode, and destination-containment checks. The original encoded
bytes are not resized or recompressed.

```bash
python scripts/download_wildfake_demo.py
python scripts/evaluate_wildfake_demo.py
```

`data/external/wildfake_demo/` contains the ignored source images, immutable
metadata copies, aggregate download manifest, and deterministic all-test
evaluation manifest. The public repository contains aggregate evidence only;
it never contains WildFake images or per-image external predictions.

The completed byte-level audit retained all exact organizer rows and found
1,808 same-label duplicate-content groups (5,124 additional rows), for 8,717
unique content hashes. No hash appeared with conflicting labels. Aggregate
metrics therefore weight the exact row set, including its disclosed repeated
content, rather than treating all 13,841 rows as statistically independent.
