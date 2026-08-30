# Submission Requirements Checklist

## Public repository

- [x] Code is pushed to the public GitHub repository.
- [ ] Setup instructions work from a fresh environment.
- [x] Training and evaluation commands are documented.
- [x] Dataset sources and licenses are documented.
- [ ] Team contributions are listed.
- [x] Limitations and future improvements are stated.

## Required inference interface

- [x] The command accepts an image directory.
- [x] Output is valid JSON.
- [x] Every record contains exactly `image_path` and `pred`.
- [x] `pred` is a number between 0 and 1 representing AIGC likelihood.
- [x] Paths are processed in deterministic order.
- [x] Corrupt or unsupported files produce clear warnings.

## Robustness evidence

- [x] Clean-image results are reported.
- [x] JPEG qualities 90, 70, 50, and 30 are evaluated.
- [x] Gaussian blur sigma 0.5, 1.0, and 2.0 is evaluated.
- [x] Downscale factors 0.5 and 0.25 are evaluated.
- [x] Gaussian noise sigma 0.02, 0.05, and 0.10 is evaluated.
- [x] Brightness, contrast, and saturation changes are evaluated.
- [x] Center crop 80% is evaluated.
- [x] A compact clean-versus-transformed table or figure is included.
- [x] Representative false positives and false negatives are included.

## Demonstration and Devpost

- [x] The demo works locally without training data.
- [ ] The Streamlit URL has been smoke-tested.
- [ ] A public YouTube demo video is available.
- [ ] The YouTube link is included in Devpost.
- [x] Development tools, model, libraries, and datasets are named.
- [ ] GitHub and optional live-demo URLs are included.

## Data isolation

- [x] The organizer WildFake demonstration subset was not used for training.
- [x] It was not used for threshold selection or model selection.
- [x] No repeated manifest source ID or normalized-image hash crosses splits.

## WildFake external demonstration evidence

- [x] The downloader pins the immutable WildFake revision and metadata hashes.
- [x] Selective HTTP ranges require valid `206 Content-Range` responses and can resume.
- [x] Extraction verifies local headers, CRC-32, size, decoding, and safe paths.
- [x] All 4,998 COCO val2017 and 8,843 Advanced DALL-E 3 images are verified locally.
- [x] The frozen checkpoint hash is verified immediately before evaluation.
- [x] Clean metrics are generated at the unchanged `0.50` threshold.
- [x] Same-label duplicate content is disclosed; conflicting labels fail closed.
- [x] Public evidence contains only aggregate metrics and no local paths.
- [x] Git contains no external images, range files, local manifests, or raw predictions.
