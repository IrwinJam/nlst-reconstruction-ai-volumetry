# Study Protocol (pre-specified 2026-09-29, before any model is trained)

**Working title:** *Same Scan, Different Answer: CT Reconstruction Changes AI Tumor Segmentation in the
National Lung Screening Trial — and Few-Shot Adaptation Can Correct It*

**Investigator:** James Irwin (independent project; public NCI data; code written new for this study).
**Target:** Stanford AI+HEALTH 2026 lightning talk (abstract due Oct 15, 2026), then a full paper.

---

## 1. Rationale
Every CT scanner reconstructs raw measurements into images with a choice of reconstruction filter
("kernel": smooth vs sharp). Hospitals and screening programs choose differently, and AI tools are rarely
validated across that choice. The National Lung Screening Trial (NLST) — the NCI trial that established
low-dose CT lung cancer screening — often stored **the same acquisition reconstructed with two different
filters**. That gives a natural controlled experiment: identical patient, tumor, scanner, dose and
moment in time; only the reconstruction differs. Any change in the AI's output is caused by
reconstruction alone.

## 2. Data (all public, NCI Imaging Data Commons, CC BY 4.0)
- **NLSTSeg**: manual expert tumor masks for 601 NLST participants (596 CT series downloadable),
  scanned at U.S. screening centers on GE (324), Siemens (187), Toshiba (43) and Philips (42) scanners.
- **Paired-reconstruction subset**: 146 participants whose annotated scan has a second reconstruction
  with identical slice thickness and slice count but a different filter (GE STANDARD→BONE 106,
  GE LUNG→STANDARD 28, Philips 14 across B/C/D filters). Mask transfer is verified slice-by-slice by
  table position before inclusion; mismatched pairs are excluded and counted.
- Files: `nlstseg_manifest.csv`, `kernel_pairs.csv`.

## 3. Aims and pre-specified endpoints

**Aim 1 — Reconstruction effect (controlled; primary).**
Apply a segmentation model to both reconstructions of the same scan and compare against the same
expert mask.
- **Primary endpoint:** per-patient paired difference in Dice between reconstructions.
- **Key secondary (clinical):** per-patient % difference in segmented tumor volume between
  reconstructions — volume drives screening follow-up decisions, so a reconstruction-only change in
  measured volume is directly clinically meaningful.

**Aim 2 — Vendor generalization (national scale).**
Leave-one-vendor-out: train on three vendors, test on the unseen fourth, for each vendor.
- Endpoint: Dice on the unseen vendor vs held-out patients from the seen vendors (same model).

**Aim 3 — Mitigation with few-shot adaptation.**
Does adapting on k annotated slices from the new patient (k = 1, 2, 4, 8) close the gaps from Aims 1–2?
Compare MAML meta-learning against a conventionally trained network fine-tuned on the same slices.
- Endpoint: fraction of the Aim 1/Aim 2 gap recovered at each k.

## 4. Analysis plan (fixed now)
- Patient-level train / validation / test split (65/10/25, seed 42) made before training; the
  146 paired-reconstruction patients are held out of training entirely and used only for Aim 1.
- Model and adaptation settings are chosen on validation patients only; the test set is scored once.
- Unit of analysis is the patient. Paired Wilcoxon signed-rank tests; patient-level bootstrap 95% CIs
  (10,000 resamples); Holm correction across the secondary endpoints.
- Robustness: 3 training seeds; report mean ± SD across seeds.
- Every negative or null result is reported alongside positive ones.

## 5. Models
- Segmentation network: 2D U-Net (GroupNorm), identical across all arms.
- Arms: (a) conventional training; (b) MAML meta-training; (c) optional reference: nnU-Net 2D, the
  field's standard supervised baseline, if compute allows before the deadline.

## 6. Timeline
| Dates | Work |
|---|---|
| Sep 29–30 | Finish NLSTSeg download; download 146 paired reconstructions (12.4 GB); verify mask transfer |
| Oct 1–5 | Train conventional + MAML models (3 seeds) |
| Oct 6–8 | Aims 1–3 evaluation, statistics, figures |
| Oct 9–11 | Abstract; independent re-check of every number against output files |
| Oct 12–15 | Buffer; submit |

## 7. Honesty guardrails
- No claim leaves this project without a traceable output file.
- The abstract reports internal validation on NLST only; it does not claim clinical performance.
- Deviations from this protocol are logged below with date and reason.

## Deviation log
- **D1 (2026-09-29, before any training) — Gate 1 mask criterion.** The pre-specified HU range
  [−300, +200] passed 466/596 masks (78%); visual review showed the misses are mainly real ground-glass
  tumors (typically −700 to −300 HU). Criterion replaced by tumor-vs-surrounding contrast ≥100 HU plus
  visual review of all flagged masks. HU-range result still reported. See `results/qc/QC_REPORT.md`.
- **D2 (2026-09-29, before any training) — Pair verification.** "Mask transfer verified slice-by-slice
  by table position" proved insufficient: one metadata-matched pair (215325) was a different
  acquisition. Added pixel-content verification (blurred-image correlation ≥0.99 on every tumor slice);
  2 pairs excluded. Final Aim 1 set: 130 series from 129 patients.
- **D3 (2026-09-30, before any test-set or Aim-1 evaluation) — Model development after Gate 2 shortfall.**
  Gate 2 failed narrowly (mean best validation Dice 0.583 vs 0.60; criterion unchanged). A
  validation-only audit (`HIGHER_DICE_PLAN.md`, `results/audit/`) showed missed tumour dominates
  (55% of error), with complete misses of large juxta-mediastinal masses and tiny nodules, and
  overfitting. Round 1 ("recipe v2") adds, and only adds: (a) a second intensity window
  (soft tissue −160…+240 HU) as input channels; (b) data augmentation (rotation ±15°, scale 0.85–1.15,
  intensity/contrast jitter, flips); (c) 100 training epochs. Continuation rule fixed now: seeds 1–2,
  MAML and leave-one-vendor-out runs proceed only if seed 0 reaches best validation Dice ≥ 0.62.
  Tuning budget: at most two rounds (Round 2 = nnU-Net 2D) before the test set is used.
  Planned secondary analysis (if Sybil boxes verify against our CT series): semi-automatic mode in
  which segmentation is restricted to a region around the independent NLST-Sybil radiologist box.
  The registered primary endpoint and fully automatic evaluation are unchanged.
- **D4 (2026-09-30, written while Round 1 seed 2 was training, before the Round 1 Gate 2 verdict) —
  Round 2 rule.** Round 2 runs only if Round 1 fails Gate 2 (mean best validation Dice of seeds 0–2
  < 0.60). Model: nnU-Net v2, `2d` configuration, default plans and preprocessing, input = the same
  tumour slabs (tumour slices ±5) in raw HU at native resolution (`code/prepare_nnunet.py`). One fold
  whose training/validation cases are exactly our locked train (290) / val (44) patients
  (`nnunet/splits_final.json`); test and Aim-1 cases are never used for training or selection.
  Budget: one run, ≤24 GPU-hours (250-epoch trainer). Gate 2 for Round 2: `checkpoint_final`
  evaluated on the 44 validation patients with our slab metric after resampling predictions to our
  1.25 mm grid; pass if mean Dice ≥ 0.60 (single seed — reported as a limitation).
  Model used for Aims 1–3: if Round 1 passes, our U-Net (as registered). If only nnU-Net passes,
  nnU-Net is the Aim 1 model and the MAML comparison (Aim 2) uses our U-Net, reported as exploratory
  with its validation Dice stated. If both fail, the budget is exhausted: the best-validating model is
  used, the Gate 2 shortfall is stated in the abstract, and no further tuning is done.
  *Outcome:* Round 1 passed Gate 2 (0.624, SD 0.010; `results/gate2_v2.md`), so Round 2 was not run.
- **D5 (2026-09-30, before any test-set or Aim-1 evaluation) — duplicated slice positions.** Found
  while exporting data for nnU-Net: a slice-spacing of 0 for two patients. A full scan
  (`code/dup_slice_check.py`, `results/qc/dup_slices.csv`; 705 annotated and paired series) found 8
  series (5 patients) storing more than one image at a table position:
  - 118672 (Aim 1): two acquisitions interleaved at every position, expert mask drawn across both →
    **excluded from Aim 1** (final Aim 1 set: 129 series from 128 patients).
  - 120849, 214975 (Aim 1): 3 and 1 duplicated positions, all outside the tumour slab; slab, spacing
    and volume verified unaffected → kept.
  - 132823 (train; exact pixel copies) and 200397 (train; three acquisitions mixed inside the slab):
    training-only label noise in 2/290 training patients. Round 1 models were already trained and are
    kept; disclosed as a limitation. No validation or test patient is affected, so Gate 2 stands.
  The nnU-Net export collapses exact copies and skips mixed-acquisition series.
- **D6 (2026-09-30, before any test-set or Aim-1 evaluation and before any box-mode code was run) —
  box-guided secondary analysis (specifies the "semi-automatic mode" planned in D3).**
  *Prompt.* NLST-Sybil radiologist boxes (independent readers, DICOM SR) are rasterized on our 1.25 mm
  grid with the same resampling as the images. Boxes are grouped into lesions (3D connected components
  of the box raster); the lesion(s) overlapping the NLSTSeg tumour form the prompt, and the prompt is
  their 3D bounding box (in-plane box + slice range). Patients without a box on the target lesion are
  not in box-mode analyses (count reported). The NLSTSeg mask is used only to pick which lesion the
  prompt refers to, as a clinician would pick the lesion to segment.
  *Model.* The Round 1 U-Net with one added input channel (prompt box on slices within its range),
  initialized from `v2_conv_s{seed}` (new channel zero-initialized), 50 epochs, lr 5e-4 cosine,
  recipe-v2 augmentation, 3 seeds. Training prompts are simulated from the expert mask's bounding box
  plus per-side offsets drawn from the empirical real-box offsets of training patients (no test, val
  or Aim-1 boxes used to calibrate). Validation uses real radiologist boxes; checkpoint = best val Dice.
  Output is restricted to the prompt box dilated 10 mm in-plane and ±1 slice (as box-prompted methods do).
  *Endpoints (secondary).* Mean slab Dice (same metric as primary) on test patients with a real prompt,
  mean ± SD over 3 seeds; comparators on the same patients: the automatic Round 1 models, and — on the
  patients that also have LIDC-annot-NLST501 reads — radiologist-vs-NLSTSeg Dice recomputed with the
  same metric and grid. Exploratory: Aim 1 volume change between paired reconstructions in box mode.
  The primary endpoint and fully automatic analyses are unchanged.
- **D7 (2026-10-02, POST-HOC — written after the frozen test/Aim-1 results were seen) — where does
  automatic volume instability come from?** Prompted by the pre-selected Fig. 1 case, in which the
  automatic change came from regions away from the tumour. For the 110 Aim-1 participants with a
  radiologist prompt, each automatic prediction (U-Net seeds 0–2; nnU-Net) is split into volume inside
  the D6 prompt region (prompt dilated 10 mm in-plane, ±1 slice) and outside it. Reported: fraction of
  patients with |ΔV| > 25% for (a) total volume and (b) inside-region volume only, and the share of the
  summed absolute volume change (mL) that lies outside the region. Labelled post-hoc everywhere; it does
  not change any pre-specified result. Script: `code/posthoc_d7.py`.
- **D8 (2026-10-03, POST-HOC — paper analyses requested by anticipated reviewers; defined before running).**
  All use existing checkpoints/predictions except (f). Script: `code/posthoc_d8.py`; outputs `results/posthoc/d8_*`.
  (a) *Same-patient comparators:* nnU-Net (automatic) and nnU-Net cropped to the D6 prompt region, scored
      on the box-mode test set (71), held-out pool (181) and radiologist subset (35).
  (b) *Automatic U-Net on the full test set:* 3-seed mean Dice on all 112 test patients, overall and by vendor.
  (c) *Volumetric agreement between reconstructions:* on log volume, d = ln V_B − ln V_A per patient
      (3-seed mean volumes for U-Net and box-guided; single run for nnU-Net; patients with a zero volume in
      either reconstruction excluded and counted). Bland–Altman bias and 95% limits of agreement,
      back-transformed to percent; repeatability coefficient RC = 2.77 × within-subject SD (wSD = SD(d)/√2),
      as percent; within-subject CV = √(exp(wSD²) − 1). For U-Net and nnU-Net on all 128, and U-Net vs
      box-guided on the 110 with prompts. Bootstrap 95% CIs (10,000 resamples).
  (d) *Tumour-size strata* (expert volume < 1, 1–5, > 5 mL): fraction with > 25% change and median absolute
      change, U-Net (3-seed) and nnU-Net.
  (e) *Prompt-selection check:* test patients with vs without a radiologist prompt — expert volume, vendor
      mix, automatic U-Net Dice (Mann–Whitney U; χ² for vendor).
  (f) *Size-matched control for the GE leave-one-vendor-out gap:* one all-vendor U-Net (seed 0, recipe v2,
      100 epochs) trained on a random, vendor-proportional subset of the training set equal in size to the
      non-GE training set (163 patients; validation set unchanged), scored on GE test patients and compared
      with the LOVO-GE and full models.
  (g) *Box-guided Dice between reconstructions:* mean difference B − A on the 110, bootstrap CI, Wilcoxon.
  None of these change pre-specified results; all are reported as post-hoc.
- **D9 (2026-10-03, POST-HOC — defined before running) — box-prompted foundation-model baseline.**
  MedSAM (wanglab/medsam-vit-base, Apache-2.0), zero-shot, no fine-tuning. Input per slice: the cached
  lung-window image (−1000…400 HU on the 1.25 mm grid) scaled to 0–255 and replicated to 3 channels;
  prompt: the same D6 radiologist box (in-plane box of the 3D prompt) on every slice inside the prompt's
  slice range; slices outside the range are predicted empty. Mask = sigmoid > 0.5, resized to the 320×320
  grid. Same slab metric. Reported on the same participants as D8(a) (test 71, held-out 181, reader 35)
  and, on the 110 Aim-1 participants with a prompt, Dice on both reconstructions, the fraction with
  > 25% volume change and the repeatability coefficient (D8c definitions). Script: `code/medsam_baseline.py`.
- **D10 (2026-10-04, POST-HOC — defined before running) — analyses for the MIDL paper.**
  (a) *Paired comparison of reproducibility:* on participants with positive volumes on both reconstructions
      for every model compared, RC (D8c definition) per model and the paired bootstrap (10,000 resamples of
      participants) CI of the RC ratio and difference: box-guided vs nnU-Net, box-guided vs MedSAM,
      box-guided vs automatic U-Net.
  (b) *MedSAM with per-slice prompts:* as D9 but each slice receives the bounding box of that slice's
      radiologist box raster (target lesion only; `rbox` in cache/NLST_v2_box); slices without a radiologist
      box are predicted empty. Same participants and outputs as D9.
  (c) *Figures:* Bland–Altman with a MedSAM panel; qualitative appendix figure of 4 test participants chosen
      by box-guided Dice at the 12.5th, 37.5th, 62.5th and 87.5th percentiles (nearest participant).
  Scripts: `code/posthoc_d10.py`, `code/medsam_baseline.py --per-slice`.
- **D11 (2026-10-05, descriptive — before any use) — cohort characteristics table.** Participant
  characteristics from the public NLST clinical tables in IDC v24 (`nlst_prsn`, `nlst_canc`): age at
  randomization, sex, smoking status and pack-years, histology and stage of the (first) lung cancer, by
  split (train / validation / test / Aim 1) plus imaging variables already in the cache (vendor, kernel,
  slice thickness, expert tumour volume). Descriptive only: no outcome is linked to these variables and no
  test is used to select anything. Script: `code/cohort_table.py`.
- **D12 (2026-10-05, POST-HOC — defined before running) — MedSAM under idealized prompts (implementation
  and metric check).** Motivated by a published report of MedSAM box-prompt Dice 0.82 on NLSTseg using 2D
  slice-level evaluation. MedSAM (as D9) with tight per-slice boxes from the expert mask's bounding box on
  each tumour slice (no margin; tumour slices only). Reported on the 71 test participants: (i) our slab Dice,
  and (ii) mean per-slice Dice over tumour slices only (closer to slice-level evaluation). Purpose:
  distinguish implementation error from differences in prompt and metric. Not used as a comparator for any
  claim. Script: `code/medsam_baseline.py --oracle`.
- **D13 (2026-10-06, POST-HOC — defined before running) — is box-guided stability more than cropping?**
  On the 110 Aim-1 participants with a prompt, the repeatability coefficient (D8c definition) of automatic
  U-Net (3-seed mean) and nnU-Net volumes restricted to the D6 prompt region (inside-region volumes from
  `results/posthoc/d7_rows.csv`), compared with the box-guided U-Net on the same participants (positive
  volumes on both reconstructions for all compared models), with paired bootstrap CIs of the RC ratio.
  Script: `code/posthoc_d13.py`.
