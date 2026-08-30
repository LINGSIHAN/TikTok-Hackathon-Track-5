# Submission Requirements Checklist

## Public repository

- [ ] Code is pushed to the public GitHub repository.
- [ ] Setup instructions work from a fresh environment.
- [ ] Training and evaluation commands are documented.
- [ ] Dataset sources and licenses are documented.
- [ ] Team contributions are listed.
- [ ] Limitations and future improvements are stated.

## Required inference interface

- [ ] The command accepts an image directory.
- [ ] Output is valid JSON.
- [ ] Every record contains exactly `image_path` and `pred`.
- [ ] `pred` is a number between 0 and 1 representing AIGC likelihood.
- [ ] Paths are processed in deterministic order.
- [ ] Corrupt or unsupported files produce clear warnings.

## Robustness evidence

- [ ] Clean-image results are reported.
- [ ] JPEG qualities 90, 70, 50, and 30 are evaluated.
- [ ] Gaussian blur sigma 0.5, 1.0, and 2.0 is evaluated.
- [ ] Downscale factors 0.5 and 0.25 are evaluated.
- [ ] Gaussian noise sigma 0.02, 0.05, and 0.10 is evaluated.
- [ ] Brightness, contrast, and saturation changes are evaluated.
- [ ] Center crop 80% is evaluated.
- [ ] A compact clean-versus-transformed table or figure is included.
- [ ] Representative false positives and false negatives are included.

## Demonstration and Devpost

- [ ] The demo works locally without training data.
- [ ] The Streamlit URL has been smoke-tested.
- [ ] A public YouTube demo video is available.
- [ ] The YouTube link is included in Devpost.
- [ ] Development tools, model, libraries, and datasets are named.
- [ ] GitHub and optional live-demo URLs are included.

## Data isolation

- [ ] The organizer WildFake demonstration subset was not used for training.
- [ ] It was not used for threshold selection or model selection.
- [ ] Generated variants of one base image do not cross dataset splits.
