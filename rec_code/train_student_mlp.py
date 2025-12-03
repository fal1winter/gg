"""
Student MLP for Recommendation
- 输入：SimCSE 语义嵌入 (1024维)
- 输出：推荐嵌入 (64维)
- 训练目标：蒸馏损失 + BPR 损失
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import time


class MLP(nn.Module):
    """多层感知机，将语义嵌入映射到推荐嵌入空间"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1, norm_type='layer'):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers - 1:  # 最后一层不加激活和 norm
                if norm_type == 'layer':
                    layers.append(nn.LayerNorm(dims[i+1]))
                elif norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class StudentModel(nn.Module):
    """学生模型：用户 MLP + 物品 MLP"""
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=64, num_layers=3, dropout=0.1):
        super(StudentModel, self).__init__()
        self.user_mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.item_mlp = MLP(input_dim, hidden_dim, output_dim, num_layers, dropout)

    def get_user_emb(self, user_feat):
        return self.user_mlp(user_feat)

    def get_item_emb(self, item_feat):
        return self.item_mlp(item_feat)

    def forward(self, user_feat, item_feat):
        user_emb = self.user_mlp(user_feat)
        item_emb = self.item_mlp(item_feat)
        return user_emb, item_emb


class BPRDataset(Dataset):
    """BPR 训练数据集"""
    def __init__(self, train_file, n_items):
        self.n_items = n_items
        self.user_pos_items = {}  # user_id -> set of positive item ids
        self.interactions = []  # (user_id, pos_item_id) pairs

        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                self.user_pos_items[user_id] = set(items)
                for item in items:
                    self.interactions.append((user_id, item))

        self.n_users = len(self.user_pos_items)
        print(f"Loaded {len(self.interactions)} interactions, {self.n_users} users, {self.n_items} items")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_item = self.interactions[idx]
        # 负采样
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.user_pos_items[user_id]:
            neg_item = np.random.randint(0, self.n_items)
        return user_id, pos_item, neg_item


def load_test_data(test_file):
    """加载测试数据"""
    test_dict = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
            test_dict[user_id] = items
    return test_dict


def load_train_pos(train_file):
    """加载训练集中的正样本（用于测试时排除）"""
    train_pos = {}
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            items = [int(x) for x in parts[1:]]
            train_pos[user_id] = set(items)
    return train_pos


def compute_metrics(rating, ground_truth, k):
    """计算 Recall@K 和 NDCG@K"""
    _, topk_indices = torch.topk(rating, k)
    topk_indices = topk_indices.cpu().numpy()

    recalls = []
    ndcgs = []

    for i, gt_items in enumerate(ground_truth):
        gt_set = set(gt_items)
        pred_items = topk_indices[i]

        # Recall@K
        hits = len(set(pred_items) & gt_set)
        recall = hits / min(len(gt_set), k) if len(gt_set) > 0 else 0
        recalls.append(recall)

        # NDCG@K
        dcg = 0.0
        for j, item in enumerate(pred_items):
            if item in gt_set:
                dcg += 1.0 / np.log2(j + 2)

        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(gt_set), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)


def test(model, user_feat, item_feat, test_dict, train_pos, topks=[20], device='cuda'):
    """测试模型"""
    model.eval()

    with torch.no_grad():
        # 获取所有嵌入（数据已在 GPU 上）
        all_user_emb = model.get_user_emb(user_feat)
        all_item_emb = model.get_item_emb(item_feat)

        results = {f'recall@{k}': 0 for k in topks}
        results.update({f'ndcg@{k}': 0 for k in topks})

        test_users = list(test_dict.keys())
        batch_size = 256

        all_recalls = {k: [] for k in topks}
        all_ndcgs = {k: [] for k in topks}

        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i:i+batch_size]
            user_emb = all_user_emb[batch_users]

            # 计算评分
            rating = torch.sigmoid(torch.matmul(user_emb, all_item_emb.t()))

            # 排除训练集中的正样本
            for j, uid in enumerate(batch_users):
                if uid in train_pos:
                    for item in train_pos[uid]:
                        if item < rating.shape[1]:
                            rating[j, item] = -1e10

            ground_truth = [test_dict[uid] for uid in batch_users]

            for k in topks:
                recall, ndcg = compute_metrics(rating, ground_truth, k)
                all_recalls[k].append(recall * len(batch_users))
                all_ndcgs[k].append(ndcg * len(batch_users))

        for k in topks:
            results[f'recall@{k}'] = sum(all_recalls[k]) / len(test_users)
            results[f'ndcg@{k}'] = sum(all_ndcgs[k]) / len(test_users)

    return results


def train_epoch(model, dataloader, user_feat, item_feat, teacher_user_emb, teacher_item_emb,
                optimizer, device, lamb_distill=0.5, lamb_bpr=0.5, use_cosine_distill=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_distill_loss = 0
    total_bpr_loss = 0

    for batch in dataloader:
        user_ids, pos_items, neg_items = batch
        user_ids = user_ids.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        # 获取输入特征（数据已在 GPU 上，直接索引）
        batch_user_feat = user_feat[user_ids]
        batch_pos_feat = item_feat[pos_items]
        batch_neg_feat = item_feat[neg_items]

        # 学生模型前向
        student_user_emb = model.get_user_emb(batch_user_feat)
        student_pos_emb = model.get_item_emb(batch_pos_feat)
        student_neg_emb = model.get_item_emb(batch_neg_feat)

        # === 蒸馏损失 ===
        teacher_user = teacher_user_emb[user_ids]
        teacher_pos = teacher_item_emb[pos_items]
        teacher_neg = teacher_item_emb[neg_items]

        if use_cosine_distill:
            # Cosine similarity loss (1 - cosine)
            distill_loss = (
                (1 - F.cosine_similarity(student_user_emb, teacher_user)).mean() +
                (1 - F.cosine_similarity(student_pos_emb, teacher_pos)).mean() +
                (1 - F.cosine_similarity(student_neg_emb, teacher_neg)).mean()
            ) / 3
        else:
            # MSE loss
            distill_loss = (
                F.mse_loss(student_user_emb, teacher_user) +
                F.mse_loss(student_pos_emb, teacher_pos) +
                F.mse_loss(student_neg_emb, teacher_neg)
            ) / 3

        # === BPR 损失 ===
        pos_scores = torch.sum(student_user_emb * student_pos_emb, dim=1)
        neg_scores = torch.sum(student_user_emb * student_neg_emb, dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # === 总损失 ===
        loss = lamb_distill * distill_loss + lamb_bpr * bpr_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_distill_loss += distill_loss.item()
        total_bpr_loss += bpr_loss.item()

    n_batches = len(dataloader)
    return total_loss / n_batches, total_distill_loss / n_batches, total_bpr_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/lastfm')
    parser.add_argument('--teacher_path', type=str, default='../result/lastfm/colakg_useragg')
    parser.add_argument('--output_path', type=str, default='../result/lastfm/student_mlp')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lamb_distill', type=float, default=0.5, help='蒸馏损失权重')
    parser.add_argument('--lamb_bpr', type=float, default=0.5, help='BPR损失权重')
    parser.add_argument('--use_cosine_distill', action='store_true', help='使用 cosine similarity 作为蒸馏损失')
    parser.add_argument('--topks', type=str, default='[20]')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 动态选择设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        # CPU 多线程优化
        num_threads = min(16, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(2, num_threads // 4))
        print(f"Using CPU (threads: {num_threads})")

    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    topks = eval(args.topks)

    print("=" * 50)
    print("Student MLP Training")
    print("=" * 50)

    # 加载数据
    print("\nLoading data...")
    user_feat = torch.load(os.path.join(args.data_path, 'lastfm_embeddings_simcse_kg_user.pt'))
    item_feat = torch.load(os.path.join(args.data_path, 'lastfm_embeddings_simcse_kg.pt'))
    teacher_user_emb = torch.load(os.path.join(args.teacher_path, 'user_embedding_final.pt'))
    teacher_item_emb = torch.load(os.path.join(args.teacher_path, 'item_embedding_final.pt'))

    print(f"User feat: {user_feat.shape}, Item feat: {item_feat.shape}")
    print(f"Teacher user emb: {teacher_user_emb.shape}, Teacher item emb: {teacher_item_emb.shape}")

    # 预先将所有数据移到 GPU（避免训练时频繁传输）
    user_feat = user_feat.to(device)
    item_feat = item_feat.to(device)
    teacher_user_emb = teacher_user_emb.to(device)
    teacher_item_emb = teacher_item_emb.to(device)
    print(f"Data moved to {device}")

    n_users, input_dim = user_feat.shape
    n_items = item_feat.shape[0]
    output_dim = teacher_user_emb.shape[1]

    # 加载训练/测试数据
    train_file = os.path.join(args.data_path, 'train.txt')
    test_file = os.path.join(args.data_path, 'test.txt')

    train_dataset = BPRDataset(train_file, n_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    test_dict = load_test_data(test_file)
    train_pos = load_train_pos(train_file)
    print(f"Test users: {len(test_dict)}")

    # 创建模型
    model = StudentModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 训练
    print("\nTraining...")
    best_ndcg = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        loss, distill_loss, bpr_loss = train_epoch(
            model, train_loader, user_feat, item_feat,
            teacher_user_emb, teacher_item_emb,
            optimizer, device,
            args.lamb_distill, args.lamb_bpr, args.use_cosine_distill
        )

        epoch_time = time.time() - start_time

        # 每 5 轮测试一次
        if epoch % 5 == 0 or epoch == 1:
            results = test(model, user_feat, item_feat, test_dict, train_pos, topks, device)
            current_ndcg = results[f'ndcg@{topks[0]}']

            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} (distill: {distill_loss:.4f}, bpr: {bpr_loss:.4f}) | "
                  f"Recall@{topks[0]}: {results[f'recall@{topks[0]}']:.4f} | "
                  f"NDCG@{topks[0]}: {current_ndcg:.4f} | Time: {epoch_time:.1f}s")

            scheduler.step(current_ndcg)

            # 保存最佳模型
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch

                # 保存嵌入
                model.eval()
                with torch.no_grad():
                    all_user_emb = model.get_user_emb(user_feat)
                    all_item_emb = model.get_item_emb(item_feat)

                    torch.save(all_user_emb.cpu(), os.path.join(args.output_path, 'user_embedding_best.pt'))
                    torch.save(all_item_emb.cpu(), os.path.join(args.output_path, 'item_embedding_best.pt'))
                    torch.save(model.state_dict(), os.path.join(args.output_path, 'model_best.pt'))

                print(f"  [BEST] Saved at epoch {epoch}, NDCG@{topks[0]}={best_ndcg:.4f}")
        else:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} (distill: {distill_loss:.4f}, bpr: {bpr_loss:.4f}) | Time: {epoch_time:.1f}s")

    print("\n" + "=" * 50)
    print(f"Training completed!")
    print(f"Best NDCG@{topks[0]}: {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Results saved to: {args.output_path}")


if __name__ == '__main__':
    main()
