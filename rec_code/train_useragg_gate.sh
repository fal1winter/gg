#!/bin/bash

# 训练 CoLaKGUserAggGate 模型
# 用户侧使用门控机制聚合交互物品嵌入
# 用户语义嵌入作为门控指导信号

DATASET="lastfm"  # ml-1m, mind, lastfm
ITEM_EMB='../data/lastfm/lastfm_embeddings_simcse_kg_l.pt'
USER_EMB='../data/lastfm/lastfm_embeddings_simcse_kg_user_l.pt'

python main.py \
    --dataset ${DATASET} \
    --model colakg_useragg_gate \
    --item_semantic_emb_file ${ITEM_EMB} \
    --user_semantic_emb_file ${USER_EMB} \
    --recdim 64 \
    --layer 3 \
    --neighbor_k 10 \
    --user_max_items 50 \
    --lr 0.003 \
    --decay 1e-4 \
    --dropout_i 0.6 \
    --dropout_u 0.6 \
    --dropout_n 0.6 \
    --bpr_batch 12048 \
    --seed 2020 \
    --comment "colakg_useragg_gate"
