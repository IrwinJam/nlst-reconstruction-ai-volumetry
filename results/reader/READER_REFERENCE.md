# Human reference: radiologist agreement (LIDC-annot-NLST501)

Source: LIDC-annot-NLST501 (TCIA, NLST collection, CC BY 4.0), 4 independent radiologist reads
(DICOM SEG) for each of 107 NLSTSeg patients; every read is on our annotated series or its verified
same-scan twin. Code: `code/reader_agreement.py` (rules fixed in its docstring). No model involved.
Run date 2026-09-30.

| Measure | Value |
|---|---|
| Reads | 428 (107 patients × 4) |
| Reads that outlined the NLSTSeg tumour | 406 / 428 (94.9%); all 4 readers in 96 / 107 patients |
| Radiologist vs radiologist, 3D Dice (592 pairs) | median **0.774**, mean 0.742 |
| Radiologist vs radiologist, per-patient mean then median | 0.761 |
| NLSTSeg vs radiologist, reads that outlined it | median **0.751**, mean 0.717 |
| NLSTSeg vs radiologist, all reads (miss = 0) | mean 0.680 |

Misses (22 reads, 11 patients): the reader's findings were 30–240 mm away from the NLSTSeg tumour (a
different lesion) or the reader outlined nothing; checked by centroid distance. **119326** (14.0 mL
NLSTSeg tumour, missed by all 4 readers, who each outlined other lesions) needs visual review.

Correction during analysis: the first run matched frames to slices by exact table position (0.01 mm)
and lost frames to ~0.01 mm rounding in the SEG files (44 reads affected, inflating misses). Matching
changed to nearest slice within 0.1 mm; zero frames now unmatched.

Caveats before comparing with the model:
- These Dice values are at native resolution over the whole series; the model metric is slab-level 3D
  Dice on the 1.25 mm grid. A like-for-like comparison must recompute reader Dice with the model metric
  on the same patients (test 16 + Aim 1 27 overlap; validation 7 were used for model selection).
- NLSTSeg is one annotation; agreement with it is not the same as accuracy.
