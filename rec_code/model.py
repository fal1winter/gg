import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)
    
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKG(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset, 
                 adj_matrix=None, 
                 semantic_emb=None, 
                 user_semantic_emb=None,):
        super(CoLaKG, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
   
        self.user_semantic_emb = user_semantic_emb.to(world.device)
        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
   
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.semantic_map = nn.Linear(1024, self.latent_dim)
        self.user_semantic_map = nn.Linear(1024, self.latent_dim)
        print(f"lgn is already to go(drop_edge:{self.config['use_drop_edge']})")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.W_u = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W_u.data, gain=1.414)
        self.a_u = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a_u.data, gain=1.414)
        self.alpha=0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        items_emb_merged = (items_emb + items_semantic_emb) / 2
        
        user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
        user_semantic_emb = self.user_semantic_map(user_semantic_emb)
        user_semantic_emb = F.elu(user_semantic_emb)
        user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
        users_emb_merged = (users_emb + user_semantic_emb) / 2
        
        
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]  # N,L,d1

        # x = self.attentions(neighbor_semantic_emb, neighbor_emb, items_semantic_emb0)
        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0
        
        Wh = torch.matmul(h, self.W)  # N,L,d
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1],1)  # N,L,d1
        Wh0 = torch.matmul(h0, self.W)  # N,L,d
        
        W_concat = torch.cat((Wh, Wh0), dim=-1) # N,L,2d
        
        attention = torch.matmul(W_concat, self.a).squeeze(-1) # N,L
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1) # N,L
    
        attention = F.dropout(attention, self.dropout_neighbor, training=self.training) # N,L
        attention = attention.unsqueeze(-1)
     
        h_prime = attention * value_emb

        h_prime = torch.sum(h_prime, dim=1)
        
        h_prime = F.elu(h_prime)
      

        items_emb_merged = (items_emb_merged + h_prime ) / 2
        
        # items_emb = F.elu(items_emb)
       
        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]
        
        if self.config['use_drop_edge']:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]
        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb_ego0

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0, users_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2) +
                         pos_emb_ego0.norm(2).pow(2) +
                         neg_emb_ego0.norm(2).pow(2) +
                         users_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKGUserAgg(BasicModel):
    """
    CoLaKG变体：用户侧不使用自己的语义嵌入，改为聚合交互过的物品语义嵌入
    """
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset,
                 adj_matrix=None,  # 物品-物品邻接矩阵
                 semantic_emb=None,  # 物品语义嵌入
                 user_item_adj=None,):  # 用户-物品邻接矩阵 (N_user, max_items)
        super(CoLaKGUserAgg, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
        self.user_item_adj = user_item_adj.to(world.device)  # 用户交互的物品索引

        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.semantic_map = nn.Linear(1024, self.latent_dim)

        # 物品邻居聚合的注意力参数
        print(f"CoLaKGUserAgg: drop_edge:{self.config['use_drop_edge']}")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for CoLaKGUserAgg
        用户侧通过聚合交互物品的语义嵌入获得表示
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # === 物品侧：语义嵌入融合 ===
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        items_emb_merged = (items_emb + items_semantic_emb) / 2

        # === 用户侧：简单均值聚合交互物品的嵌入 ===
        # user_item_adj: (N_user, max_items) 每个用户交互的物品索引
        user_item_value = items_emb_merged[self.user_item_adj]  # (N_user, max_items, latent_dim)

        # 计算 mask（非零位置表示有效交互）
        mask = (self.user_item_adj > 0).float().unsqueeze(-1)  # (N_user, max_items, 1)
        mask_sum = mask.sum(dim=1).clamp(min=1)  # (N_user, 1) 防止除零

        # 均值聚合
        user_semantic_agg = (user_item_value * mask).sum(dim=1) / mask_sum  # (N_user, latent_dim)
        user_semantic_agg = F.dropout(user_semantic_agg, self.dropout_u, training=self.training)

        users_emb_merged = (users_emb + user_semantic_agg) / 2

        # === 物品侧：邻居聚合 ===
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]  # N,L,d1

        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0

        Wh = torch.matmul(h, self.W)  # N,L,d
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1], 1)  # N,L,d1
        Wh0 = torch.matmul(h0, self.W)  # N,L,d

        W_concat = torch.cat((Wh, Wh0), dim=-1)  # N,L,2d

        attention = torch.matmul(W_concat, self.a).squeeze(-1)  # N,L
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1)  # N,L

        attention = F.dropout(attention, self.dropout_neighbor, training=self.training)
        attention = attention.unsqueeze(-1)

        h_prime = attention * value_emb
        h_prime = torch.sum(h_prime, dim=1)
        h_prime = F.elu(h_prime)

        items_emb_merged = (items_emb_merged + h_prime) / 2

        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]

        if self.config['use_drop_edge']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        # 用 users_emb 代替原来的 users_emb_ego0，避免重复计算聚合
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0, posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0, users_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2) +
                         pos_emb_ego0.norm(2).pow(2) +
                         neg_emb_ego0.norm(2).pow(2) +
                         users_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKGUserAggSSL(CoLaKGUserAgg):
    """
    CoLaKGUserAgg + SSL对比学习
    继承 CoLaKGUserAgg，只重写 bpr_loss 添加对比学习损失
    """
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0, posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0, users_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2) +
                         pos_emb_ego0.norm(2).pow(2) +
                         neg_emb_ego0.norm(2).pow(2) +
                         users_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # === SSL 对比学习（随机负采样版本）===
        ssl_temp = 0.4  # 温度系数
        ssl_reg = 0.3   # 对齐损失的权重
        num_neg = 256   # 负例采样数量
        eps = 0.3

        batch_size = posEmb0.shape[0]

        # 加噪声
        random_noise_id = torch.rand_like(posEmb0).to(world.device)
        posEmb0_noise = posEmb0 + torch.sign(posEmb0) * F.normalize(random_noise_id, dim=1) * eps
        random_noise_sem = torch.rand_like(pos_emb_ego0).to(world.device)
        pos_emb_ego0_noise = pos_emb_ego0 + torch.sign(pos_emb_ego0) * F.normalize(random_noise_sem, dim=1) * eps

        # 归一化
        view_id = F.normalize(posEmb0, dim=1)
        view_sem = F.normalize(pos_emb_ego0, dim=1)
        view_id_noise = F.normalize(posEmb0_noise, dim=1)
        view_sem_noise = F.normalize(pos_emb_ego0_noise, dim=1)

        # 随机采样负例索引
        neg_indices = torch.randint(0, batch_size, (batch_size, num_neg), device=world.device)

        # 正例得分
        pos_score = (view_id_noise * view_sem).sum(dim=1) + (view_id * view_sem_noise).sum(dim=1)
        pos_score = pos_score.unsqueeze(1)

        # 负例得分
        neg_sem = view_sem[neg_indices]
        neg_sem_noise = view_sem_noise[neg_indices]
        neg_score = torch.bmm(neg_sem, view_id_noise.unsqueeze(-1)).squeeze(-1) + \
                    torch.bmm(neg_sem_noise, view_id.unsqueeze(-1)).squeeze(-1)

        # InfoNCE loss
        logits = torch.cat([pos_score, neg_score], dim=1) / ssl_temp
        labels = torch.zeros(batch_size, dtype=torch.long, device=world.device)
        loss_ssl = F.cross_entropy(logits, labels)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        loss = loss + ssl_reg * loss_ssl

        return loss, reg_loss


class CoLaKGUserAggAtt(BasicModel):
    """
    CoLaKG变体：用户侧使用注意力机制聚合交互过的物品语义嵌入
    注意：计算量较大，训练速度较慢
    """
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset,
                 adj_matrix=None,  # 物品-物品邻接矩阵
                 semantic_emb=None,  # 物品语义嵌入
                 user_item_adj=None,):  # 用户-物品邻接矩阵 (N_user, max_items)
        super(CoLaKGUserAggAtt, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
        self.user_item_adj = user_item_adj.to(world.device)

        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.semantic_map = nn.Linear(1024, self.latent_dim)

        # 物品邻居聚合的注意力参数
        print(f"CoLaKGUserAggAtt: drop_edge:{self.config['use_drop_edge']}")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 用户侧聚合物品的注意力参数
        self.W_u = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W_u.data, gain=1.414)
        self.a_u = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a_u.data, gain=1.414)

        # 用户查询向量，用于聚合物品
        self.user_query = nn.Linear(self.latent_dim, 32)

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for CoLaKGUserAggAtt
        用户侧使用注意力机制聚合交互物品的语义嵌入
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # === 物品侧：语义嵌入融合 ===
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        items_emb_merged = (items_emb + items_semantic_emb) / 2

        # === 用户侧：注意力聚合交互物品的语义嵌入 ===
        user_item_semantic = self.semantic_emb[self.user_item_adj]  # (N_user, max_items, 1024)
        user_item_value = items_emb_merged[self.user_item_adj]  # (N_user, max_items, latent_dim)

        # 使用用户ID嵌入作为query
        user_query = self.user_query(users_emb)  # (N_user, 32)

        # 注意力计算
        Wh = torch.matmul(user_item_semantic, self.W_u)  # (N_user, max_items, 32)
        user_query_expanded = user_query.unsqueeze(1).expand(-1, Wh.shape[1], -1)  # (N_user, max_items, 32)

        W_concat = torch.cat((Wh, user_query_expanded), dim=-1)  # (N_user, max_items, 64)
        attention = torch.matmul(W_concat, self.a_u).squeeze(-1)  # (N_user, max_items)
        attention = self.leakyrelu(attention)

        # mask padding位置
        mask = (self.user_item_adj > 0).float()
        attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout_u, training=self.training)
        attention = attention.unsqueeze(-1)  # (N_user, max_items, 1)

        # 聚合物品嵌入
        user_semantic_agg = torch.sum(attention * user_item_value, dim=1)  # (N_user, latent_dim)
        user_semantic_agg = F.elu(user_semantic_agg)

        users_emb_merged = (users_emb + user_semantic_agg) / 2

        # === 物品侧：邻居聚合 ===
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]

        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0

        Wh = torch.matmul(h, self.W)
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1], 1)
        Wh0 = torch.matmul(h0, self.W)

        W_concat = torch.cat((Wh, Wh0), dim=-1)

        attention = torch.matmul(W_concat, self.a).squeeze(-1)
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout_neighbor, training=self.training)
        attention = attention.unsqueeze(-1)

        h_prime = attention * value_emb
        h_prime = torch.sum(h_prime, dim=1)
        h_prime = F.elu(h_prime)

        items_emb_merged = (items_emb_merged + h_prime) / 2

        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]

        if self.config['use_drop_edge']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0, posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2) +
                         pos_emb_ego0.norm(2).pow(2) +
                         neg_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class CoLaKGUserAggGate(BasicModel):
    """
    CoLaKG变体：用户侧使用门控机制聚合交互物品嵌入
    使用用户语义嵌入作为门控指导信号
    """
    def __init__(self,
                 config:dict,
                 dataset:BasicDataset,
                 adj_matrix=None,  # 物品-物品邻接矩阵
                 semantic_emb=None,  # 物品语义嵌入
                 user_item_adj=None,  # 用户-物品邻接矩阵
                 user_semantic_emb=None,):  # 用户语义嵌入（门控指导）
        super(CoLaKGUserAggGate, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        self.semantic_emb = semantic_emb.to(world.device)
        self.user_item_adj = user_item_adj.to(world.device)
        self.user_semantic_emb = user_semantic_emb.to(world.device)

        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        print("self.num_items", self.num_items)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.semantic_map = nn.Linear(1024, self.latent_dim)
        self.user_semantic_map = nn.Linear(1024, self.latent_dim)

        # 物品邻居聚合的注意力参数
        print(f"CoLaKGUserAggGate: drop_edge:{self.config['use_drop_edge']}")
        self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 门控网络：用用户语义和物品语义拼接后计算门控值
        # gate = sigmoid(MLP([user_sem; item_sem]))
        gate_hidden = 256
        self.gate_net = nn.Sequential(
            nn.Linear(1024 * 2, gate_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_u),
            nn.Linear(gate_hidden, self.latent_dim)
        )

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for CoLaKGUserAggGate
        用户侧使用门控机制聚合交互物品嵌入，用户语义嵌入作为门控指导
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        # === 物品侧：语义嵌入融合 ===
        items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
        items_semantic_emb = self.semantic_map(items_semantic_emb)
        items_semantic_emb = F.elu(items_semantic_emb)
        items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
        items_emb_merged = (items_emb + items_semantic_emb) / 2

        # === 用户侧：门控聚合交互物品嵌入 ===
        # user_item_adj: (N_user, max_items)
        user_item_semantic = self.semantic_emb[self.user_item_adj]  # (N_user, max_items, 1024)
        user_item_value = items_emb_merged[self.user_item_adj]  # (N_user, max_items, latent_dim)

        # 用户语义嵌入作为门控指导
        user_sem = self.user_semantic_emb  # (N_user, 1024)
        user_sem = F.dropout(user_sem, self.dropout_u, training=self.training)

        # 计算门控值：gate = sigmoid(MLP([user_sem; item_sem]))
        # 扩展用户语义到每个物品位置
        max_items = user_item_semantic.shape[1]
        user_sem_expanded = user_sem.unsqueeze(1).expand(-1, max_items, -1)  # (N_user, max_items, 1024)
        # 拼接用户语义和物品语义
        concat_sem = torch.cat([user_sem_expanded, user_item_semantic], dim=-1)  # (N_user, max_items, 2048)
        # 门控值（通过 2 层 MLP）
        gate = torch.sigmoid(self.gate_net(concat_sem))  # (N_user, max_items, latent_dim)
        gate = F.dropout(gate, self.dropout_u, training=self.training)

        # 门控加权
        gated_items = gate * user_item_value  # (N_user, max_items, latent_dim)

        # mask padding 并均值聚合
        mask = (self.user_item_adj > 0).float().unsqueeze(-1)  # (N_user, max_items, 1)
        mask_sum = mask.sum(dim=1).clamp(min=1)  # (N_user, 1)
        user_semantic_agg = (gated_items * mask).sum(dim=1) / mask_sum  # (N_user, latent_dim)
        user_semantic_agg = F.elu(user_semantic_agg)

        users_emb_merged = (users_emb + user_semantic_agg) / 2

        # === 物品侧：邻居聚合（与其他模型相同）===
        neighbor_emb = items_emb_merged[self.adj_matrix]
        items_semantic_emb0 = self.semantic_emb
        neighbor_semantic_emb = self.semantic_emb[self.adj_matrix]

        h, value_emb, semantic_emb = neighbor_semantic_emb, neighbor_emb, items_semantic_emb0

        Wh = torch.matmul(h, self.W)
        h0 = semantic_emb.unsqueeze(1).repeat(1, h.shape[1], 1)
        Wh0 = torch.matmul(h0, self.W)

        W_concat = torch.cat((Wh, Wh0), dim=-1)

        attention = torch.matmul(W_concat, self.a).squeeze(-1)
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout_neighbor, training=self.training)
        attention = attention.unsqueeze(-1)

        h_prime = attention * value_emb
        h_prime = torch.sum(h_prime, dim=1)
        h_prime = F.elu(h_prime)

        items_emb_merged = (items_emb_merged + h_prime) / 2

        all_emb = torch.cat([users_emb_merged, items_emb_merged])
        embs = [all_emb]

        if self.config['use_drop_edge']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        users_emb_ego0 = self.user_semantic_map(self.user_semantic_emb)[users]
        pos_emb_ego0 = self.semantic_map(self.semantic_emb)[pos_items]
        neg_emb_ego0 = self.semantic_map(self.semantic_emb)[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, pos_emb_ego0, neg_emb_ego0, users_emb_ego0

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0, posEmb0, negEmb0, pos_emb_ego0, neg_emb_ego0, users_emb_ego0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2) +
                         pos_emb_ego0.norm(2).pow(2) +
                         neg_emb_ego0.norm(2).pow(2) +
                         users_emb_ego0.norm(2).pow(2)
                         )/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
