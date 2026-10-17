#!/usr/bin/env bash
# round 1 queue (D3): unet + maml seeds then leave one vendor out
cd /c/MetaSeg_data
export PYTHONIOENCODING=utf-8
CACHE='C:\MetaSeg_data\cache\NLST_v2'
LOG=results/round1.log
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
run() {
  local name=$1; shift
  for attempt in 1 2 3; do
    say "START $name (attempt $attempt)"
    if python code/nlst_seg.py "$@" --aug --cache "$CACHE" >> results/train_queue_v2.log 2>&1; then say "OK    $name"; return 0; fi
    say "FAIL  $name (attempt $attempt) - retrying in 60 s"; sleep 60
  done
  say "GAVE UP on $name after 3 attempts"
}

say "Round 1 waiting for two-window preprocessing"
until grep -q "^cached 575" results/prepare_v2.log 2>/dev/null; do sleep 30; done
say "preprocessing done: $(grep '^cached\|resampling' results/prepare_v2.log | tr '\n' ' ')"

run v2_conv_s0 --stage conv --run v2_conv_s0 --seed 0 --epochs 100
best0=$(python - <<'PY'
import re
t = "\n".join(l for l in open(r"C:\MetaSeg_data\results\v2_conv_s0\log.txt", encoding="utf-8") if "resume" not in l)
v = [float(x) for x in re.findall(r"val Dice ([0-9.]+)", t)]
print(max(v) if v else 0)
PY
)
say "v2 seed 0 best validation Dice = $best0 (continuation rule: >= 0.62)"
# continuation rule from D3, seed 0 has to hit 0.62 or we stop
if python -c "import sys; sys.exit(0 if float('$best0') >= 0.62 else 1)"; then
  say "CONTINUE: rule met"
  for s in 1 2; do run v2_conv_s$s --stage conv --run v2_conv_s$s --seed $s --epochs 100; done
  python code/gate2_report.py v2_ >> "$LOG" 2>&1
  for s in 0 1 2; do run v2_maml_s$s --stage maml --run v2_maml_s$s --seed $s --meta-iters 4000 --init results/v2_conv_s$s/conv_best.pt; done
  for v in "GE MEDICAL SYSTEMS" "SIEMENS" "TOSHIBA" "Philips"; do
    tag="v2_lovo_${v// /_}"
    run "$tag conv" --stage conv --run "$tag" --seed 0 --epochs 100 --exclude-vendor "$v"
    run "$tag maml" --stage maml --run "$tag" --seed 0 --meta-iters 4000 --exclude-vendor "$v" --init "results/$tag/conv_best.pt"
  done
else
  say "STOP: rule not met, per D3 no more round 1 runs. next is round 2 (nnU-Net 2D) after review"
fi
python code/gate2_report.py v2_ >> "$LOG" 2>&1
say "ROUND 1 QUEUE COMPLETE"
touch results/OVERNIGHT_DONE
