#!/usr/bin/env bash
# train the box guided unet, seeds 0-2 (D6)
cd /c/MetaSeg_data
export PYTHONIOENCODING=utf-8
LOG=results/box.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "waiting for Round 1 queue (results/OVERNIGHT_DONE)"
until [ -f results/OVERNIGHT_DONE ]; do sleep 60; done
for s in 0 1 2; do
  for attempt in 1 2 3; do
    say "START v2_box_s$s (attempt $attempt)"
    if python code/box_mode.py --run v2_box_s$s --seed $s >> results/box_train.log 2>&1; then say "OK    v2_box_s$s"; break; fi
    say "FAIL  v2_box_s$s (attempt $attempt) - retrying in 60 s"; sleep 60
  done
done
touch results/BOX_DONE
say "box queue complete"
