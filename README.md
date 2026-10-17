# NLST reconstruction study

This repository holds the code and results for a study of AI lung tumor segmentation on CT scans from the National Lung Screening Trial (NLST). The study measures how much the tumor volume reported by AI models changes when the same scan is reconstructed with a different filter. It also tests whether a radiologist box prompt makes that volume more stable.

## Data

All data are public and come from the NCI Imaging Data Commons (IDC).

- NLST CT scans (doi:10.7937/TCIA.HMQ8-J677)
- NLSTSeg expert tumor masks (doi:10.5281/zenodo.17362624)
- NLST-Sybil radiologist lesion boxes (doi:10.5281/zenodo.15643335)
- LIDC-annot-NLST501 multi-reader segmentations (doi:10.7937/Y2DX-E334)
- NLST clinical tables from IDC release v24

Raw images and preprocessed caches are not stored here. The scripts in `code/` download and rebuild them. `nlstseg_manifest.csv` lists the annotated series and `kernel_pairs.csv` lists the candidate paired reconstructions.

## Protocol

`STUDY_PROTOCOL.md` was written before any model was trained. It defines the aims, end points, patient split, statistics and quality gates. Every later change is recorded there as a numbered deviation (D1 to D12) with its date, and each was committed before the analysis it describes was run. The test set and the paired-reconstruction cohort were scored once by `code/evaluate.py` at commit `d8df62c`.

## Layout

| Path | Contents |
|---|---|
| `code/` | Download, quality control, preprocessing, training, evaluation and figure scripts |
| `results/qc/` | Data-integrity reports and exclusion lists |
| `results/split.json` | Locked patient split |
| `results/eval/` | Frozen evaluation outputs (`summary.json` and per-patient CSV files) |
| `results/posthoc/` | Post hoc analyses D7 to D12 |
| `results/reader/` | Radiologist agreement analysis |
| `results/cohort/` | Participant characteristics table |
| `results/figures/` | Figures |

## Reproducing the results

Run the steps in order from the repository root with Python 3.11. The GPU steps used one 8 GB NVIDIA card.

1. Download the data: `code/download_nlstseg.py`, `code/finish_download.py`, `code/download_pairs.py`, `code/download_lidc501_seg.py`.
2. Check data integrity: `code/qc_gate1.py`, `code/qc_pair_content.py`, `code/dup_slice_check.py`.
3. Build the training cache: `code/prepare_nlst.py` with `NLST_TWO_WINDOWS=1` and `NLST_CACHE_OUT=cache/NLST_v2`.
4. Train the U-Net and MAML models, including the leave-one-vendor-out runs: `code/run_round1.sh`.
5. Train the box-guided model: `code/prepare_boxes.py`, then `code/run_box.sh`.
6. Train nnU-Net: `code/prepare_nnunet.py`, then `code/run_nnunet.sh` and `code/run_after_nnunet.sh`.
7. Score the test set and paired reconstructions: `code/evaluate.py`.
8. Run the post hoc analyses: `code/posthoc_d7.py`, `code/posthoc_d8.py`, `code/train_sizematched.py`, `code/medsam_baseline.py`, `code/posthoc_d10.py`, `code/medsam_oracle.py`. The MedSAM scripts run in a separate environment with Hugging Face Transformers.
9. Build the tables and figures: `code/cohort_table.py`, `code/reader_agreement.py`, `code/make_figures.py`, `code/make_paper_figures.py`, `code/make_supplement.py`.
10. Check that every number in the manuscript traces to a result file: `code/verify_paper.py`. The manuscript is not part of this repository until publication.

## Main results

Changing the reconstruction filter did not change Dice, but AI tumor volume changed by more than 25% in 24% of participants with the U-Net and in 15% with nnU-Net. The box-guided model had a repeatability coefficient of 12% across reconstructions, compared with 90% for nnU-Net on the same participants.
