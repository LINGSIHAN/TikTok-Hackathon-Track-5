# Data Pipeline Handoff

## Files Created
- `src/data/dataset.py`
- `src/data/__init__.py`
- `scripts/prepare_sid_subset.py`
- `tests/data/test_dataset.py`
- `data/README.md`

## Public API / Exports
- `ImageManifestDataset(manifest_path, split, transform, root_dir)`
- `compute_sha256(file_path)`

## Test Execution
Run unit tests with pytest:
```bash
pytest tests/data/test_dataset.py