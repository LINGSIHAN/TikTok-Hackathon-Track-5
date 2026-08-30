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
