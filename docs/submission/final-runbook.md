# Deadline Runbook

This is the shortest safe route from the current repository to submission. Do
not collect more data, retrain, change architectures, or tune against test or
WildFake results.

## Completed: calibration and model decision

The single validation-only calibration run completed without retraining. It
locked threshold `0.40042864`, then exploratorily re-scored the previously
observed SID and GenImage tests exactly once. Four of eight deployment gates
failed: SID false-positive rate, clean balanced accuracy, mean transformed
balanced accuracy, and worst transformed balanced accuracy.

The final decision is **retain v1 at threshold `0.50`**. Do not rerun
calibration, retrain, or promote the v2 checkpoint. Reviewed aggregate evidence
is in [`genimage_v2_calibration.json`](../../artifacts/metrics/genimage_v2_calibration.json)
and [`genimage-v2-calibration-report.md`](genimage-v2-calibration-report.md).

## 1. Verify and deploy

Local release verification passed: 253 tests passed (one platform-specific
symlink test skipped), the real v1 checkpoint completed CPU inference, the
required directory-to-JSON command produced valid output, and the Streamlit
entrypoint loaded without exceptions. Then:

1. Push the reviewed final commit to `master`.
2. In Streamlit Community Cloud, deploy or reboot:
   - repository: `LINGSIHAN/TikTok-Hackathon-Track-5`;
   - branch: `master`;
   - entrypoint: `app/streamlit_app.py`;
   - secrets: none.
3. Test one authentic and one generated image and run the robustness passport
   for both.
4. Save the exact public URL.

## 2. Record the demo

Follow `docs/submission/demo-script.md` and keep the video to 2–3 minutes. Show:

- one authentic and one generated prediction;
- the robustness passport;
- the required batch JSON output;
- the strongest comparison figure and the honest calibration decision;
- limitations; and
- the GitHub and Streamlit URLs.

Upload to YouTube with public visibility and verify playback while signed out.

## 3. Finish Devpost

Use `docs/submission/devpost-draft.md`, then:

1. replace every `Name TBD` contribution row;
2. add the GitHub, Streamlit, and YouTube URLs;
3. add the strongest two or three figures;
4. preview every link; and
5. submit with at least 30 minutes left for upload or form issues.

Leave any unverified item in `docs/submission/requirements-checklist.md`
unchecked rather than claiming it was completed.
