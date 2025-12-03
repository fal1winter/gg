#!/bin/bash

# Student MLP 训练脚本

cd /home/sun/pythoncode/CoLaKG-SIGIR25/rec_code

python train_student_mlp.py \
    --data_path ../data/lastfm \
    --teacher_path ../result/lastfm/colakg_useragg \
    --output_path ../result/lastfm/student_mlp \
    --hidden_dim 512 \
    --output_dim 64 \
    --num_layers 3 \
    --dropout 0.1 \
    --lr 3e-3 \
    --weight_decay 1e-4 \
    --batch_size 12048 \
    --epochs 1000 \
    --lamb_distill 0 \
    --lamb_bpr 1 \
    --topks "[20]" \
    --device cuda \
    --seed 2024
