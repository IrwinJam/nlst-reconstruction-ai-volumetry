# Gate 1: Data Integrity Report

**Date:** 2026-09-29 · **Verdict: PASS, with two documented protocol deviations (D1, D2)**
Scripts: `code/qc_gate1.py`, `code/qc_pair_content.py`, `code/qc_review_montage.py`
All per-patient results: `qc_masks.csv`, `qc_pairs.csv`, `qc_pairs_content.csv`, `final_status.csv`,
`aim1_pairs_final.csv`, `exclusions.csv`, `manual_review.csv`.

## Summary
| | Result | Gate criterion | Status |
|---|---|---|---|
| Usable expert masks | **575 / 601 (95.7%)** | ≥ 95% (after D1) | ✅ |
| Aim 1 paired reconstructions | **130 series / 129 patients** | ≥ 100 patients | ✅ |
| Mask frames on CT slice grid (included patients) | max offset **0.001 mm** | ≤ 0.01 mm | ✅ |
| Pair content agreement (included pairs) | min correlation **0.9952** | ≥ 0.99 | ✅ |

## 1. Expert masks (601)
Each mask was matched to its CT slices and checked three ways: (a) the pre-specified HU range,
(b) tumor-vs-surrounding-lung contrast, (c) slice-grid alignment. Flagged cases were reviewed visually.

| Outcome | n | Reason |
|---|---|---|
| Usable | 575 | NA |
| Excluded | 19 | Mask drawn on a different slice grid (frames 0.06–0.90 mm off the CT slices; 3 could not be mapped at all). Strict 0.01 mm tolerance, set before inspection, kept. GE 11, Siemens 8 |
| Excluded | 5 | CT series never published in IDC |
| Excluded | 1 | A mask frame references a CT slice that does not exist |
| Excluded | 1 | Visual review: outline on chest wall with no clearly visible nodule (118553) |

**Visual review:** all 8 low-contrast flags were inspected (`review_flagged.png`); 7 retained with a
visible explanation (pure ground-glass, cavitation, mass abutting mediastinum, diaphragm in the ring),
1 excluded. Twelve random "ground-glass" passes and 24 random patients across all vendors
(`montage_masks.png`) all showed outlines correctly placed. **Review decisions are provisional
(confirmed by the investigator on 2026-10-06).**

## 2. Paired reconstructions
| Step | Series remaining |
|---|---|
| Candidate pairs (same thickness & slice count, different filter) | 148 |
| Annotated mask usable | 132 |
| Identical slice positions, spacing, origin, orientation, matrix | (13 metadata failures excluded earlier) |
| **Pixel-content check** (D2): blurred-image correlation ≥ 0.99 on every tumor slice | **130** |

Content-check exclusions: **215325**: different acquisition despite matching metadata (anatomy shifted;
median correlation 0.92; `outlier_215325.png`); **200397**: two consecutive corrupted slices in the
paired series (correlation 0.84–0.85, all other slices 1.0000). Both annotated masks remain valid, so
these patients return to the general pool for Aims 2–3.

Final Aim 1 set by filter pair: GE STANDARD→BONE 92 · GE LUNG→STANDARD 25 · Philips D→C 8 ·
Philips B→D 4 · Philips B→C 1. Visual check: `montage_pairs.png`.

## 3. Descriptive finding (no model involved)
Reconstruction alone shifts the measured mean density inside the expert tumor outline. Median
absolute shift **15.4 HU** (IQR 5.8–29.8; max 194) across the 130 pairs; by filter pair: LUNG→STANDARD
median −65.8 HU, Philips D→C −43.4, B→D +57.6, STANDARD→BONE +9.8 (`hu_shift_by_pair.csv`).
This is context for Aim 1, not an endpoint.

## Deviations (also logged in STUDY_PROTOCOL.md)
- **D1: HU-range criterion replaced.** The pre-specified rule (≥95% of masks with mean HU in
  [−300, +200]) passed only 466/596 (78%). Visual review showed the misses are overwhelmingly real
  ground-glass tumors, which normally read −700 to −300 HU. Replaced by tumor-vs-surround contrast
  ≥100 HU (min among usable masks: 104 HU) plus visual review of every flagged mask. The HU-range
  result is still reported.
- **D2: Pixel-content verification added for pairs.** Metadata matching accepted a pair that was a
  different acquisition (215325). Added the correlation check above; 2 pairs excluded.

## Next (Day 3)
Preprocessing must use only `final_status.csv` = PASS patients and `aim1_pairs_final.csv` pairs;
splits are locked after preprocessing, before any training.
