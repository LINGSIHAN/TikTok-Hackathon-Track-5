# Deadline Runbook

This is the shortest safe route from the current repository to submission. Do
not collect more data, retrain, change architectures, or tune against test or
WildFake results.

## 1. Lock the lowest-safe SID false-positive threshold

Run `notebooks/calibrate_genimage_v2_kaggle.ipynb`, not either training
notebook. The model weights remain unchanged.

The predeclared validation-only rule is:

1. Enumerate every attainable decision threshold from the SID and GenImage
   clean validation scores.
2. Keep only thresholds with SID generated-image recall at least 95%, GenImage
   generated-image recall at least 55%, GenImage balanced accuracy at least
   73%, and GenImage balanced accuracy within two percentage points of its
   validation optimum.
3. Select the remaining threshold with the fewest SID false positives.
4. Lock it, then apply it once to the saved SID and GenImage test predictions.
   Those tests were already reviewed at 0.50 and are therefore an exploratory
   deployment check, not a fresh unbiased holdout. Never retune from them.

This prevents a meaningless zero-false-positive result obtained by labelling
every image authentic.

### Kaggle clicks

1. After the calibration code has been committed and pushed, create a private
   Kaggle dataset containing the original file named exactly
   `genimage_v2_export.zip`.
2. Import `notebooks/calibrate_genimage_v2_kaggle.ipynb` into Kaggle.
3. In **Settings**, enable **Internet** and select a **T4 GPU**.
4. Under **Add Input**, attach:
   - `cartografia/unbiased-tiny-genimage` version 1; and
   - the private input containing `genimage_v2_export.zip`.
5. Choose **Run All** once. Do not run the v2 training notebook again.
6. Wait for `Validated calibration export ready`.
7. Download `/kaggle/working/genimage_v2_calibration_export.zip`.
8. Save it locally under `artifacts/runs/kaggle/` and ask Codex to review it.

Timebox this phase to 60–90 minutes. The notebook deliberately uses a fresh
calibration checkout so new tracked evidence cannot collide with generated
files in the original training checkout. It recreates the deterministic
prepared images but still performs no training; a typical fresh run should take
roughly 20–45 minutes. The Output GiB counter is disk usage, not progress.

## 2. Make exactly one deployment decision

Treat v2 as deployment-eligible only if the exploratory locked-threshold
re-score passes every gate:

- SID false-positive rate at most 5%;
- SID balanced accuracy at least 93%;
- SID generated-image recall at least 90%; and
- GenImage balanced accuracy at least 73%;
- SID mean/worst transformed balanced accuracy at least 90%/85%; and
- GenImage mean/worst transformed balanced accuracy at least 73%/65%.

If any gate fails, retain v1 and its 0.50 threshold. Do not launch a second
calibration attempt. Codex should perform the checkpoint/threshold promotion,
lineage update, evidence import, and regression tests after reviewing the ZIP.

## 3. Verify and deploy

After the model decision, Codex runs the full tests, CPU inference, required
directory-to-JSON command, and local Streamlit smoke test. Then:

1. Push the reviewed final commit to `master`.
2. In Streamlit Community Cloud, deploy or reboot:
   - repository: `LINGSIHAN/TikTok-Hackathon-Track-5`;
   - branch: `master`;
   - entrypoint: `app/streamlit_app.py`;
   - secrets: none.
3. Test one authentic and one generated image and run the robustness passport
   for both.
4. Save the exact public URL.

## 4. Record the demo

Follow `docs/submission/demo-script.md` and keep the video to 2–3 minutes. Show:

- one authentic and one generated prediction;
- the robustness passport;
- the required batch JSON output;
- the strongest comparison figure and the honest calibration decision;
- limitations; and
- the GitHub and Streamlit URLs.

Upload to YouTube with public visibility and verify playback while signed out.

## 5. Finish Devpost

Use `docs/submission/devpost-draft.md`, then:

1. replace every `Name TBD` contribution row;
2. add the GitHub, Streamlit, and YouTube URLs;
3. add the strongest two or three figures;
4. preview every link; and
5. submit with at least 30 minutes left for upload or form issues.

Leave any unverified item in `docs/submission/requirements-checklist.md`
unchecked rather than claiming it was completed.
