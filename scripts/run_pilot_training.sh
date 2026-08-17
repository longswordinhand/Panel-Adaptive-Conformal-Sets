#!/usr/bin/env bash
set -euo pipefail

TAG="main"
EPOCHS=60
PYTHON="/home/yguo56/miniforge3/envs/vmunet_gen/bin/python"

cleanup() {
  status=$?
  echo "RUNNER_EXIT status=${status}"
  exit "$status"
}
trap cleanup EXIT INT TERM

for fold in 0 1 2 3 4; do
  for task in 01 02; do
    model="experiments/pilot/models/${TAG}/fold${fold}/task${task}/model.pt"
    meta="experiments/pilot/predictions/${TAG}/fold${fold}/task${task}/metadata.json"
    if [[ -s "$model" && -s "$meta" ]]; then
      count=$(find "experiments/pilot/predictions/${TAG}/fold${fold}/task${task}" -maxdepth 1 -name 'case*.npy' -type f | wc -l)
      if [[ "$count" -eq 22 ]]; then
        echo "SKIP fold=${fold} task=${task} complete"
        continue
      fi
    fi
    echo "START fold=${fold} task=${task} epochs=${EPOCHS}"
    "$PYTHON" scripts/train_pilot_unet.py --fold "$fold" --task "$task" --epochs "$EPOCHS" --tag "$TAG"
    echo "DONE fold=${fold} task=${task}"
  done
done

echo "ALL_TRAINING_COMPLETE"
