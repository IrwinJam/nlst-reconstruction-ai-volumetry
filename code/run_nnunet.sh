#!/usr/bin/env bash
# train nnunet 2d on our locked split
cd /c/MetaSeg_data
export nnUNet_raw='C:\MetaSeg_data\nnunet\raw' nnUNet_preprocessed='C:\MetaSeg_data\nnunet\preprocessed' \
       nnUNet_results='C:\MetaSeg_data\nnunet\results' nnUNet_compile=f nnUNet_n_proc_DA=4 PYTHONIOENCODING=utf-8
TR=nnUNetTrainer_250epochs
LOG=results/nnunet_run.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "waiting for box-mode queue (results/BOX_DONE)"
until [ -f results/BOX_DONE ]; do sleep 60; done
say "GPU free; starting nnU-Net training ($TR, 2d, fold 0 = our split)"
ok=0
for attempt in 1 2 3; do
  flag=""; [ $attempt -gt 1 ] && flag="--c"
  say "START train (attempt $attempt)"
  if nnUNetv2_train 501 2d 0 -tr $TR $flag >> results/nnunet_train.log 2>&1; then ok=1; say "OK    train"; break; fi
  say "FAIL  train (attempt $attempt) - retrying in 60 s"; sleep 60
done
if [ $ok = 1 ]; then
  say "START predict held-out cases"
  if nnUNetv2_predict -i "$nnUNet_raw\\Dataset501_NLSTSeg\\imagesTs" -o 'C:\MetaSeg_data\nnunet\pred_ts' \
       -d 501 -c 2d -f 0 -tr $TR -chk checkpoint_final.pth >> results/nnunet_predict.log 2>&1; then
    say "OK    predict ($(ls nnunet/pred_ts/*.nii.gz 2>/dev/null | wc -l) files)"
  else say "FAIL  predict"; fi
else
  say "GAVE UP on training after 3 attempts"
fi
touch nnunet/NNUNET_DONE
say "nnU-Net queue complete"
