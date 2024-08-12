#!/usr/bin/env bash

seeds="42"
gpu='0'


config='config_AASCE'
for seed in $seeds
do
  default_command="--seed ${seed} --config ${config}"
  custom_command=""
  CUDA_VISIBLE_DEVICES="${gpu}" python -u main.py ${default_command} ${custom_command} --save_test_prediction --subpixel_inference 15 --use_prev_heatmap_only_for_hint_index
done


