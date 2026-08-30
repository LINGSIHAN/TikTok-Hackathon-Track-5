# Data Pipeline Handoff

## Public interfaces

- `scripts/prepare_sid_subset.py --total 6000 --seed 42` streams a small,
  balanced subset from `saberzl/SID_Set` and writes the images, manifest, and
  run summary under `data/`.
- `prepare_examples(...)` accepts an iterable of SID-like records for direct
  tests without network access.
- `ImageManifestDataset(manifest_path, split, transform, root_dir)` validates
  the manifest and yields `(image, label)` in manifest order.
- `compute_sha256(file_path)` computes a chunked file digest.

## Important semantics

- Labels are binary: `0` is real and `1` is fully synthetic; label `2` is
  excluded.
- Local splits are named `train`, `val`, and `test`.
- Hashes of the stored normalized JPEGs are unique, and a SID `img_id` cannot
  cross splits.
- Both classes are normalized through the same JPEG export path to reduce
  trivial file-format shortcuts.
- An empty or under-filled source fails loudly instead of producing a manifest
  that would break later during training.

Dataset provenance and license notes are in `data/README.md`.

## Validation

```bash
pytest tests/data
```
