#!/bin/bash

dataset="mimic_cxr"
annotation="/data2/yuhaowang/MIMIC-CXR/mimic_annotation_all.json"
base_dir="/data2/yuhaowang/MIMIC-CXR/files"
delta_file="/home/yuhaowang/project/report_generation/TRRG/R2GenGPT/deep_checkpoint_step42310.pth"

version="v1_deep"
savepath="./save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt
