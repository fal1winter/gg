import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import datetime
import os
from os.path import join
import register
from register import dataset
from sklearn.metrics.pairwise import cosine_similarity

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
k = world.config['neighbor_k']

log_file = f"../logs/{world.dataset}_{world.model_name}_neighbor{str(k)}_{current_time}.txt"

item_semantic_emb = torch.load(world.item_semantic_emb_file)
user_semantic_emb = torch.load(world.user_semantic_emb_file)
cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())
sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)
sorted_indices = sorted_indices[:, 1:k+1] # does not include itself
sorted_indices = torch.tensor(sorted_indices).long()


def build_user_item_adj(dataset, max_items=50):
    """
    构建用户-物品邻接矩阵
    Args:
        dataset: 数据集对象，需要有 allPos 属性
        max_items: 每个用户最多保留的交互物品数
    Returns:
        user_item_adj: (N_users, max_items) 的 LongTensor
    """
    all_pos = dataset.allPos  # list of arrays, allPos[u] = items interacted by user u
    n_users = dataset.n_users

    user_item_adj = torch.zeros(n_users, max_items, dtype=torch.long)

    for user_id in range(n_users):
        items = all_pos[user_id]
        if len(items) > 0:
            # 如果物品数超过 max_items，随机采样
            if len(items) > max_items:
                selected = np.random.choice(items, max_items, replace=False)
            else:
                selected = items
            user_item_adj[user_id, :len(selected)] = torch.tensor(selected)

    return user_item_adj


# 根据模型类型选择不同的初始化方式
if world.model_name in ['colakg_useragg', 'colakg_useragg_ssl', 'colakg_useragg_att']:
    # CoLaKGUserAgg 系列: 使用用户-物品邻接矩阵，不使用用户语义嵌入
    user_item_adj = build_user_item_adj(dataset, max_items=world.config.get('user_max_items', 50))
    print(f"Built user_item_adj with shape: {user_item_adj.shape}")
    Recmodel = register.MODELS[world.model_name](
        world.config, dataset, sorted_indices, item_semantic_emb, user_item_adj
    )
elif world.model_name == 'colakg_useragg_gate':
    # CoLaKGUserAggGate: 使用用户-物品邻接矩阵 + 用户语义嵌入作为门控指导
    user_item_adj = build_user_item_adj(dataset, max_items=world.config.get('user_max_items', 50))
    print(f"Built user_item_adj with shape: {user_item_adj.shape}")
    Recmodel = register.MODELS[world.model_name](
        world.config, dataset, sorted_indices, item_semantic_emb, user_item_adj, user_semantic_emb
    )
else:
    # CoLaKG 等其他模型：使用用户语义嵌入
    Recmodel = register.MODELS[world.model_name](
        world.config, dataset, sorted_indices, item_semantic_emb, user_semantic_emb
    )

Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
    

with open(log_file, "w") as f:
    f.write("Training Log\n")
    f.write("====================\n")

# 创建 result 目录
result_dir = join(world.ROOT_PATH, 'result', world.dataset, world.model_name)
os.makedirs(result_dir, exist_ok=True)

def save_embeddings(model, epoch_num, ndcg_value):
    """保存最终用于推荐的嵌入"""
    model.eval()
    with torch.no_grad():
        all_users_emb, all_items_emb = model.computer()

        user_emb_path = join(result_dir, 'user_embedding_best.pt')
        item_emb_path = join(result_dir, 'item_embedding_best.pt')

        torch.save(all_users_emb.cpu(), user_emb_path)
        torch.save(all_items_emb.cpu(), item_emb_path)

        print(f"[BEST] Embeddings saved at epoch {epoch_num}, NDCG={ndcg_value:.4f}")
        print(f"  User: {all_users_emb.shape}, Item: {all_items_emb.shape}")
    model.train()

# 跟踪最佳 NDCG
best_ndcg = 0.0
best_epoch = 0

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        if epoch % 5 == 0:
            cprint("[TEST]")
            # 强制启用多核测试
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, multicore=1)
            log_message = f'TEST RESULTS at EPOCH[{epoch+1}/{world.TRAIN_epochs}]: {test_results}'
            print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message + "\n")

            # 检查是否是最佳 NDCG，如果是则保存嵌入
            current_ndcg = test_results['ndcg'][0]  # NDCG@topks[0]
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                save_embeddings(Recmodel, epoch + 1, current_ndcg)

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)

        end = time.time()
        epoch_time = end - start

        log_message = f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} - Time: {epoch_time:.2f} seconds'
        print(log_message)

        with open(log_file, "a") as f:
            f.write(log_message + "\n")

        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()

print(f"Training completed! Best NDCG={best_ndcg:.4f} at epoch {best_epoch}")
print(f"Best embeddings saved to: {result_dir}")
