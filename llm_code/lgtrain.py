import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
from torch.utils.data import Dataset, DataLoader
import time
torch.set_num_threads(64)
# ================= 配置参数 =================
CONFIG = {
    'user_emb_path': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg_user.pt',
    'item_emb_path': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg.pt',
    'train_file': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/train.txt',

    'epochs': 300,           # 要求的训练轮数
    'batch_size': 9192,      # 根据显存调整
    'lr': 0.001,             # 学习率
    'n_layers': 3,           # LightGCN 层数
    'reg_weight': 1e-4,      # L2 正则化权重
    'diff_weight': 0.05,     # 差异损失权重 (用余弦相似度，只约束方向)
    'var_weight': 0.2,       # 方差最大化损失权重 (鼓励embedding分散)
    'uniformity_weight': 0.05,  # 均匀性损失权重 (让embedding空间分布更均匀)
    'decorr_weight': 0.01,   # 去相关损失权重 (减少维度间相关性)
    'user_var_boost': 2.0,   # User方差权重放大倍数 (因为user变化较小)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on device: {CONFIG['device']}")

# ================= 数据处理 =================

class GraphDataset(Dataset):
    def __init__(self, train_file, num_users, num_items):
        # 假设 train.txt 格式为: user_id item_id (每行一对)
        # 分隔符可能是空格或制表符
        data = []
        with open(train_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                cols = line.strip().split()
                u, i = int(cols[0]), int(cols[1])
                data.append([u, i])
        
        self.train_data = np.array(data)
        self.num_users = num_users
        self.num_items = num_items
        
        # 构建用于 BPR 采样的字典
        self.user_pos_items = {}
        for u, i in self.train_data:
            if u not in self.user_pos_items:
                self.user_pos_items[u] = []
            self.user_pos_items[u].append(i)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # BPR 三元组采样: (user, pos_item, neg_item)
        user = self.train_data[idx][0]
        pos_item = self.train_data[idx][1]
        
        # 随机采样负例
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in self.user_pos_items.get(user, []):
                break
                
        return user, pos_item, neg_item

def build_adj_matrix(train_data, num_users, num_items):
    print("Building adjacency matrix...")
    # 构造 LightGCN 的邻接矩阵: 
    # A = [0, R]
    #     [R.T, 0]
    R = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    for u, i in train_data:
        R[u, i] = 1.
    R = R.tolil()
    
    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    
    # 填充矩阵
    adj_mat[:num_users, num_users:] = R
    adj_mat[num_users:, :num_users] = R.T
    
    # 归一化: D^-0.5 * A * D^-0.5
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    # 转为 PyTorch Sparse Tensor
    vals = norm_adj.data
    indices = np.vstack((norm_adj.row, norm_adj.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vals)
    shape = norm_adj.shape
    
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

# ================= 模型定义 =================

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, init_user_emb, init_item_emb, config):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = config['n_layers']
        self.device = config['device']
        self.config = config  # 保存config引用

        # 使用加载的 Tensor 初始化 Embedding，允许微调 (freeze=False)
        self.user_embedding = nn.Embedding.from_pretrained(init_user_emb, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(init_item_emb, freeze=False)

        # 保存初始权重的副本用于计算 Difference Loss (固定不更新)
        self.fixed_init_user_emb = init_user_emb.clone().detach().to(self.device)
        self.fixed_init_item_emb = init_item_emb.clone().detach().to(self.device)

    def forward(self, adj):
        # 图卷积传播
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        final_user, final_item = torch.split(light_out, [self.num_users, self.num_items])
        return final_user, final_item

    def uniformity_loss(self, x, t=2):
        """
        均匀性损失: 鼓励embedding在空间中均匀分布
        L = log(mean(exp(-t * ||x_i - x_j||^2)))
        采样计算以节省内存
        """
        # 随机采样一部分进行计算 (避免 O(n^2) 计算)
        batch_size = min(512, x.shape[0])
        indices = torch.randperm(x.shape[0])[:batch_size]
        x_sample = x[indices]
        x_sample = F.normalize(x_sample, dim=1)  # L2归一化到单位球面

        # 计算两两距离
        sq_pdist = torch.pdist(x_sample, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def decorrelation_loss(self, x):
        """
        去相关损失: 减少embedding各维度之间的相关性
        最小化非对角线元素的相关系数
        """
        # 中心化
        x_centered = x - x.mean(dim=0, keepdim=True)
        # 计算协方差矩阵
        cov = torch.mm(x_centered.T, x_centered) / (x.shape[0] - 1)
        # 标准化得到相关系数矩阵
        std = x.std(dim=0, keepdim=True).T
        std_matrix = torch.mm(std, std.T) + 1e-8
        corr = cov / std_matrix
        # 最小化非对角线元素的平方和
        mask = 1 - torch.eye(corr.shape[0], device=corr.device)
        return (corr * mask).pow(2).mean()

    def calculate_loss(self, users, pos_items, neg_items, final_user_emb, final_item_emb):
        # 1. 获取 batch 对应的 embedding
        u_emb = final_user_emb[users]
        pos_emb = final_item_emb[pos_items]
        neg_emb = final_item_emb[neg_items]

        # 2. BPR Loss
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

        # 3. Regularization Loss (L2 on current 0-layer weights)
        u_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)
        reg_loss = (1/2) * (u_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) / float(len(users))

        # 4. Difference Loss - 改用余弦相似度 (只约束方向，不约束模长)
        # 这样给方差更大的变化空间
        user_cos_sim = F.cosine_similarity(self.user_embedding.weight, self.fixed_init_user_emb, dim=1).mean()
        item_cos_sim = F.cosine_similarity(self.item_embedding.weight, self.fixed_init_item_emb, dim=1).mean()
        diff_loss = 2 - user_cos_sim - item_cos_sim  # 最大化相似度 = 最小化 (2 - sim)

        # 5. Variance Maximization Loss (负值，因为要最大化方差)
        user_var = self.user_embedding.weight.var(dim=0).mean()
        item_var = self.item_embedding.weight.var(dim=0).mean()
        # User 方差给更大权重
        user_var_boost = self.config.get('user_var_boost', 1.0)
        var_loss = -(user_var_boost * user_var + item_var)

        # 6. Uniformity Loss - 鼓励embedding均匀分布
        uniform_user = self.uniformity_loss(self.user_embedding.weight)
        uniform_item = self.uniformity_loss(self.item_embedding.weight)
        uniform_loss = uniform_user + uniform_item

        # 7. Decorrelation Loss - 减少维度间相关性
        decorr_user = self.decorrelation_loss(self.user_embedding.weight)
        decorr_item = self.decorrelation_loss(self.item_embedding.weight)
        decorr_loss = decorr_user + decorr_item

        return bpr_loss, reg_loss, diff_loss, var_loss, uniform_loss, decorr_loss

# ================= 主程序 =================

def main():
    # 1. 加载初始 Embeddings
    print("Loading initial embeddings...")
    init_user_emb = torch.load(CONFIG['user_emb_path'], map_location='cpu')
    init_item_emb = torch.load(CONFIG['item_emb_path'], map_location='cpu')
    
    num_users = init_user_emb.shape[0]
    num_items = init_item_emb.shape[0]
    print(f"Users: {num_users}, Items: {num_items}, Dim: {init_user_emb.shape[1]}")

    # 2. 准备数据
    dataset = GraphDataset(CONFIG['train_file'], num_users, num_items)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    
    # 3. 构建图
    adj_matrix = build_adj_matrix(dataset.train_data, num_users, num_items)
    adj_matrix = adj_matrix.to(CONFIG['device'])
    
    # 4. 初始化模型
    model = LightGCN(num_users, num_items, init_user_emb, init_item_emb, CONFIG).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 5. 训练循环
    print(f"Start training for {CONFIG['epochs']} epochs...")
    model.train()
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        t0 = time.time()
        
        for users, pos_items, neg_items in dataloader:
            users = users.to(CONFIG['device'])
            pos_items = pos_items.to(CONFIG['device'])
            neg_items = neg_items.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # 前向传播
            final_user_emb, final_item_emb = model(adj_matrix)
            
            # 计算 Loss
            bpr, reg, diff, var, uniform, decorr = model.calculate_loss(users, pos_items, neg_items, final_user_emb, final_item_emb)

            loss = (bpr
                    + CONFIG['reg_weight'] * reg
                    + CONFIG['diff_weight'] * diff
                    + CONFIG['var_weight'] * var
                    + CONFIG['uniformity_weight'] * uniform
                    + CONFIG['decorr_weight'] * decorr)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss:.4f} | Time: {time.time()-t0:.2f}s")

    # 6. 保存结果
    print("Training finished. Saving embeddings...")
    
    # 获取最终训练好的 0-layer embedding (基础语义 + 微调) 
    # 如果你需要保存传播后的 embedding，请改为 model(adj_matrix) 的返回值
    # 这里根据要求保存 "训练后的结果"，通常指微调后的 embedding lookup table
    final_user_weights = model.user_embedding.weight.data.cpu()
    final_item_weights = model.item_embedding.weight.data.cpu()
    
    # 构造保存路径
    user_save_path = CONFIG['user_emb_path'].replace('.pt', '_l.pt')
    item_save_path = CONFIG['item_emb_path'].replace('.pt', '_l.pt')
    
    torch.save(final_user_weights, user_save_path)
    torch.save(final_item_weights, item_save_path)
    
    print(f"Saved user embeddings to: {user_save_path}")
    print(f"Saved item embeddings to: {item_save_path}")

if __name__ == '__main__':
    main()