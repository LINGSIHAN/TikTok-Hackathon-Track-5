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
