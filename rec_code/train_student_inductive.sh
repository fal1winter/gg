#!/bin/bash

# Student MLP 归纳式训练脚本
# 测试时包含训练时未见过的用户（冷启动）

cd /home/sun/pythoncode/CoLaKG-SIGIR25/rec_code

python train_student_mlp_inductive.py \
    --data_path ../data/lastfm \
    --teacher_path ../result/lastfm/colakg_useragg \
    --output_path ../result/lastfm/student_mlp_inductive \
    --hidden_dim 512 \
    --output_dim 64 \
    --num_layers 3 \
    --dropout 0.1 \
    --lr 3e-3 \
    --weight_decay 1e-4 \
    --batch_size 2048 \
    --epochs 1000 \
    --lamb_distill 0.5 \
    --lamb_bpr 0.5 \
    --unseen_ratio 0.2 \
    --topks "[20]" \
    --seed 2024
