"""
Student MLP for Recommendation - Inductive Version
归纳式学习：测试时包含训练时未见过的用户/物品

关键区别：
- 训练集只包含 seen_users/seen_items
- 测试集包含 unseen_users（冷启动用户）
- MLP 只用语义嵌入作为输入，可以泛化到新用户/新物品
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
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1, norm_type='layer'):
        super(MLP, self).__init__()
        self.num_layers = num_layers

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers - 1:
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


class BPRDatasetInductive(Dataset):
    """归纳式 BPR 训练数据集 - 只包含 seen users 的交互"""
    def __init__(self, train_file, seen_users, n_items):
        self.n_items = n_items
        self.seen_users = set(seen_users)
        self.user_pos_items = {}
        self.interactions = []

        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                user_id = int(parts[0])
                # 只保留 seen users 的交互
                if user_id not in self.seen_users:
                    continue
                items = [int(x) for x in parts[1:]]
                self.user_pos_items[user_id] = set(items)
                for item in items:
                    self.interactions.append((user_id, item))

        print(f"Inductive training: {len(self.interactions)} interactions from {len(self.user_pos_items)} seen users")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_item = self.interactions[idx]
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.user_pos_items[user_id]:
            neg_item = np.random.randint(0, self.n_items)
        return user_id, pos_item, neg_item


def load_all_data(train_file, test_file):
    """加载所有用户的训练和测试数据"""
    train_dict = {}
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                train_dict[user_id] = set(items)

    test_dict = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                items = [int(x) for x in parts[1:]]
                test_dict[user_id] = items

    return train_dict, test_dict


def split_users_inductive(train_dict, test_dict, unseen_ratio=0.2, seed=2024):
    """
    划分用户为 seen 和 unseen
    - seen_users: 训练时可见，用于训练 MLP
    - unseen_users: 训练时不可见，只用于测试（冷启动用户）
    """
    np.random.seed(seed)

    # 只考虑在测试集中有数据的用户
    all_users = list(test_dict.keys())
    np.random.shuffle(all_users)

    n_unseen = int(len(all_users) * unseen_ratio)
    unseen_users = set(all_users[:n_unseen])
    seen_users = set(all_users[n_unseen:])

    print(f"User split: {len(seen_users)} seen, {len(unseen_users)} unseen ({unseen_ratio*100:.0f}%)")

    return seen_users, unseen_users


def compute_metrics(rating, ground_truth, k):
    """计算 Recall@K 和 NDCG@K"""
    _, topk_indices = torch.topk(rating, k)
    topk_indices = topk_indices.cpu().numpy()

    recalls = []
    ndcgs = []

    for i, gt_items in enumerate(ground_truth):
        gt_set = set(gt_items)
        pred_items = topk_indices[i]

        hits = len(set(pred_items) & gt_set)
        recall = hits / min(len(gt_set), k) if len(gt_set) > 0 else 0
        recalls.append(recall)

        dcg = 0.0
        for j, item in enumerate(pred_items):
            if item in gt_set:
                dcg += 1.0 / np.log2(j + 2)
        idcg = sum(1.0 / np.log2(j + 2) for j in range(min(len(gt_set), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return np.mean(recalls), np.mean(ndcgs)


def test_inductive(model, user_feat, item_feat, test_dict, train_dict,
                   seen_users, unseen_users, topks=[20], device='cpu'):
    """
    归纳式测试
    分别报告 seen users 和 unseen users 的性能
    """
    model.eval()

    with torch.no_grad():
        all_user_emb = model.get_user_emb(user_feat)
        all_item_emb = model.get_item_emb(item_feat)

        results = {
            'seen': {f'recall@{k}': 0 for k in topks},
            'unseen': {f'recall@{k}': 0 for k in topks},
            'all': {f'recall@{k}': 0 for k in topks}
        }
        for key in results:
            results[key].update({f'ndcg@{k}': 0 for k in topks})

        batch_size = 256

        for user_type, user_set in [('seen', seen_users), ('unseen', unseen_users)]:
            test_users = [u for u in user_set if u in test_dict]
            if len(test_users) == 0:
                continue

            all_recalls = {k: [] for k in topks}
            all_ndcgs = {k: [] for k in topks}

            for i in range(0, len(test_users), batch_size):
                batch_users = test_users[i:i+batch_size]
                user_emb = all_user_emb[batch_users]

                rating = torch.sigmoid(torch.matmul(user_emb, all_item_emb.t()))

                # 排除训练集中的正样本
                for j, uid in enumerate(batch_users):
                    if uid in train_dict:
                        for item in train_dict[uid]:
                            if item < rating.shape[1]:
                                rating[j, item] = -1e10

                ground_truth = [test_dict[uid] for uid in batch_users]

                for k in topks:
                    recall, ndcg = compute_metrics(rating, ground_truth, k)
                    all_recalls[k].append(recall * len(batch_users))
                    all_ndcgs[k].append(ndcg * len(batch_users))

            for k in topks:
                results[user_type][f'recall@{k}'] = sum(all_recalls[k]) / len(test_users)
                results[user_type][f'ndcg@{k}'] = sum(all_ndcgs[k]) / len(test_users)

        # 计算整体性能
        all_test_users = list(test_dict.keys())
        n_seen = len([u for u in seen_users if u in test_dict])
        n_unseen = len([u for u in unseen_users if u in test_dict])
        n_total = n_seen + n_unseen

        for k in topks:
            results['all'][f'recall@{k}'] = (
                results['seen'][f'recall@{k}'] * n_seen +
                results['unseen'][f'recall@{k}'] * n_unseen
            ) / n_total if n_total > 0 else 0

            results['all'][f'ndcg@{k}'] = (
                results['seen'][f'ndcg@{k}'] * n_seen +
                results['unseen'][f'ndcg@{k}'] * n_unseen
            ) / n_total if n_total > 0 else 0

    return results


def train_epoch(model, dataloader, user_feat, item_feat, teacher_user_emb, teacher_item_emb,
                optimizer, device, lamb_distill=0.5, lamb_bpr=0.5, seen_users=None):
    """训练一个 epoch（只用 seen users）"""
    model.train()
    total_loss = 0
    total_distill_loss = 0
    total_bpr_loss = 0

    for batch in dataloader:
        user_ids, pos_items, neg_items = batch
        user_ids = user_ids.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        batch_user_feat = user_feat[user_ids]
        batch_pos_feat = item_feat[pos_items]
        batch_neg_feat = item_feat[neg_items]

        student_user_emb = model.get_user_emb(batch_user_feat)
        student_pos_emb = model.get_item_emb(batch_pos_feat)
        student_neg_emb = model.get_item_emb(batch_neg_feat)

        # 蒸馏损失（只对 seen users 有教师嵌入）
        distill_loss = torch.tensor(0.0, device=device)
        if lamb_distill > 0 and teacher_user_emb is not None:
            teacher_user = teacher_user_emb[user_ids]
            teacher_pos = teacher_item_emb[pos_items]
            teacher_neg = teacher_item_emb[neg_items]

            distill_loss = (
                F.mse_loss(student_user_emb, teacher_user) +
                F.mse_loss(student_pos_emb, teacher_pos) +
                F.mse_loss(student_neg_emb, teacher_neg)
            ) / 3

        # BPR 损失
        pos_scores = torch.sum(student_user_emb * student_pos_emb, dim=1)
        neg_scores = torch.sum(student_user_emb * student_neg_emb, dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

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
    parser.add_argument('--output_path', type=str, default='../result/lastfm/student_mlp_inductive')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lamb_distill', type=float, default=0.5)
    parser.add_argument('--lamb_bpr', type=float, default=0.5)
    parser.add_argument('--unseen_ratio', type=float, default=0.2, help='比例的用户作为 unseen（冷启动）')
    parser.add_argument('--topks', type=str, default='[20]')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 动态选择设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        num_threads = min(16, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        print(f"Using CPU (threads: {num_threads})")

    topks = eval(args.topks)

    print("=" * 60)
    print("Student MLP Training - INDUCTIVE VERSION")
    print("=" * 60)

    # 加载数据
    print("\nLoading data...")
    user_feat = torch.load(os.path.join(args.data_path, 'lastfm_embeddings_simcse_kg_user.pt'))
    item_feat = torch.load(os.path.join(args.data_path, 'lastfm_embeddings_simcse_kg.pt'))

    # 教师嵌入（可选）
    teacher_user_emb = None
    teacher_item_emb = None
    if args.lamb_distill > 0:
        try:
            teacher_user_emb = torch.load(os.path.join(args.teacher_path, 'user_embedding_final.pt'))
            teacher_item_emb = torch.load(os.path.join(args.teacher_path, 'item_embedding_final.pt'))
            print(f"Teacher embeddings loaded: user {teacher_user_emb.shape}, item {teacher_item_emb.shape}")
        except:
            print("Warning: Teacher embeddings not found, using pure BPR")
            args.lamb_distill = 0
            args.lamb_bpr = 1

    print(f"User feat: {user_feat.shape}, Item feat: {item_feat.shape}")

    # 移到设备
    user_feat = user_feat.to(device)
    item_feat = item_feat.to(device)
    if teacher_user_emb is not None:
        teacher_user_emb = teacher_user_emb.to(device)
        teacher_item_emb = teacher_item_emb.to(device)

    n_users, input_dim = user_feat.shape
    n_items = item_feat.shape[0]

    # 加载训练/测试数据
    train_file = os.path.join(args.data_path, 'train.txt')
    test_file = os.path.join(args.data_path, 'test.txt')
    train_dict, test_dict = load_all_data(train_file, test_file)

    # 划分 seen/unseen 用户
    seen_users, unseen_users = split_users_inductive(
        train_dict, test_dict, args.unseen_ratio, args.seed
    )

    # 创建只包含 seen users 的训练数据
    train_dataset = BPRDatasetInductive(train_file, seen_users, n_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 创建模型
    output_dim = teacher_user_emb.shape[1] if teacher_user_emb is not None else args.output_dim
    model = StudentModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    os.makedirs(args.output_path, exist_ok=True)

    # 训练
    print("\nTraining (inductive setting)...")
    print(f"  lamb_distill={args.lamb_distill}, lamb_bpr={args.lamb_bpr}")
    print(f"  unseen_ratio={args.unseen_ratio}")
    print()

    best_ndcg_unseen = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        loss, distill_loss, bpr_loss = train_epoch(
            model, train_loader, user_feat, item_feat,
            teacher_user_emb, teacher_item_emb,
            optimizer, device,
            args.lamb_distill, args.lamb_bpr, seen_users
        )

        epoch_time = time.time() - start_time

        if epoch % 5 == 0 or epoch == 1:
            results = test_inductive(
                model, user_feat, item_feat, test_dict, train_dict,
                seen_users, unseen_users, topks, device
            )

            k = topks[0]
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")
            print(f"  SEEN   - Recall@{k}: {results['seen'][f'recall@{k}']:.4f}, NDCG@{k}: {results['seen'][f'ndcg@{k}']:.4f}")
            print(f"  UNSEEN - Recall@{k}: {results['unseen'][f'recall@{k}']:.4f}, NDCG@{k}: {results['unseen'][f'ndcg@{k}']:.4f}")
            print(f"  ALL    - Recall@{k}: {results['all'][f'recall@{k}']:.4f}, NDCG@{k}: {results['all'][f'ndcg@{k}']:.4f}")

            current_ndcg_unseen = results['unseen'][f'ndcg@{k}']
            scheduler.step(current_ndcg_unseen)

            # 保存最佳模型（基于 unseen users 的性能）
            if current_ndcg_unseen > best_ndcg_unseen:
                best_ndcg_unseen = current_ndcg_unseen
                best_epoch = epoch

                model.eval()
                with torch.no_grad():
                    all_user_emb = model.get_user_emb(user_feat)
                    all_item_emb = model.get_item_emb(item_feat)

                    torch.save(all_user_emb.cpu(), os.path.join(args.output_path, 'user_embedding_best.pt'))
                    torch.save(all_item_emb.cpu(), os.path.join(args.output_path, 'item_embedding_best.pt'))
                    torch.save(model.state_dict(), os.path.join(args.output_path, 'model_best.pt'))

                print(f"  [BEST UNSEEN] Saved at epoch {epoch}")
            print()
        else:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {epoch_time:.1f}s")

    print("=" * 60)
    print(f"Training completed!")
    print(f"Best UNSEEN NDCG@{topks[0]}: {best_ndcg_unseen:.4f} at epoch {best_epoch}")
    print(f"Results saved to: {args.output_path}")


if __name__ == '__main__':
    main()
