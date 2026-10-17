#!/usr/bin/env bash
# predict held out cases with nnunet then run the gpu eval sections
cd /c/MetaSeg_data
export nnUNet_raw='C:\MetaSeg_data\nnunet\raw' nnUNet_preprocessed='C:\MetaSeg_data\nnunet\preprocessed' \
       nnUNet_results='C:\MetaSeg_data\nnunet\results' nnUNet_compile=f PYTHONIOENCODING=utf-8
TR=nnUNetTrainer_250epochs
FOLD=nnunet/results/Dataset501_NLSTSeg/${TR}__nnUNetPlans__2d/fold_0
LOG=results/nnunet_run.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "post-training chain waiting for $FOLD/checkpoint_final.pth and trainer exit"
until [ -f "$FOLD/checkpoint_final.pth" ] && ! ps -ef | grep -v grep | grep -q nnUNetv2_train; do sleep 60; done
# wait for the cpu eval to finish so they dont fight over ram
until grep -q "CPU SECTIONS COMPLETE" results/eval/eval.log 2>/dev/null; do sleep 60; done
say "START predict held-out cases"
if nnUNetv2_predict -i "$nnUNet_raw\\Dataset501_NLSTSeg\\imagesTs" -o 'C:\MetaSeg_data\nnunet\pred_ts' \
     -d 501 -c 2d -f 0 -tr $TR -chk checkpoint_final.pth >> results/nnunet_predict.log 2>&1; then
  say "OK    predict ($(ls nnunet/pred_ts/*.nii.gz 2>/dev/null | wc -l) files)"
else say "FAIL  predict"; fi
say "START evaluation sections aim3,nnunet (GPU)"
python code/evaluate.py --sections aim3,nnunet >> results/eval/eval_gpu.log 2>&1 && say "OK    evaluation aim3,nnunet" || say "FAIL  evaluation aim3,nnunet"
touch nnunet/NNUNET_DONE
say "post-training chain complete"
