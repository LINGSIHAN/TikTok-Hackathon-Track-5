# Transformations handoff

## Files created

- `src/transforms/__init__.py`
- `src/transforms/robustness.py`
- `scripts/render_transform_contact_sheet.py`
- `tests/transforms/test_robustness.py`

## Public API

- `TRANSFORM_GRID`: exact published evaluation severities.
- `apply_transform(image, transform_name, severity, seed=42)`: returns a same-size RGB PIL image.
- `sample_training_transform(image, seed, clean_probability=0.35)`: deterministically returns a clean or transformed image plus metadata.

## Exact semantics

- JPEG uses Pillow baseline JPEG encode/decode with the specified quality.
- Gaussian blur uses Pillow's `GaussianBlur(radius=sigma)`.
- Resize downsamples with Lanczos to the selected scale, then restores the original size with Lanczos.
- Gaussian noise adds a seeded normal sample on floating-point pixels in `[0, 1]`, then clips and converts to `uint8`.
- Brightness, contrast, and saturation use factors of `1 + severity`; `-0.2` and `+0.2` therefore mean 80% and 120% of the original factor.
- Center crop retains the central 80% of both dimensions, then restores the original size with Lanczos.

Every operation converts inputs to RGB and preserves the source dimensions. The contact-sheet script writes only its explicitly requested output path.

## Dependencies

Only existing project dependencies are used: Pillow and NumPy.

## Commands and tests

```bash
python -m pytest tests/transforms
python scripts/render_transform_contact_sheet.py --input example.jpg --output /tmp/transforms.png
```

## Limitations

The published prompt does not define JPEG chroma subsampling, blur-kernel sizing, or resize interpolation. This implementation fixes those choices for repeatability and records them above. Test coverage uses synthetic images and does not establish model robustness.
