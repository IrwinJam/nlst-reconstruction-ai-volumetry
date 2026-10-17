# Figure rules (fixed 2026-10-02 before any example case was viewed)

## Fig 1: example "same scan, two filters" case
- Pool: Aim-1 participants with a radiologist prompt (box_rows.csv, set == aim1; n = 110).
- Eligible: automatic U-Net volume change |V_B - V_A| / V_A > 25% (3-seed mean per patient,
  aim1_per_seed.csv).
- Pick: sort eligible by |change|; take the lower median. No other criterion, no swapping.
- Display: slice with the largest expert-mask area; 100 mm window centred on the mask; contours from
  seed-0 models (automatic U-Net and box-guided U-Net); volumes printed are 3-seed means from the CSVs.
- The rule and the chosen patient ID are reported in the caption.

## Other figures
- Plotted values come only from results/eval/*.csv / summary.json; per-patient values are 3-seed means.
