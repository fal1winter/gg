import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import argparse

# 默认路径配置
DEFAULT_CONFIG = {
    'kg_l': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg_l.pt',
    'kg': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg.pt',
    'kg_user_l': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg_user_l.pt',
    'kg_user': '/home/sun/pythoncode/CoLaKG-SIGIR25/data/lastfm/lastfm_embeddings_simcse_kg_user.pt',
}

def load_embedding(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def compare_embeddings(emb_l, emb_orig, name):
    """比较训练后和原始embedding的相似度和方差"""
    print(f"\n【{name}】")
    print("-" * 50)

    # 余弦相似度
    cos_sim = cosine_similarity(emb_l, emb_orig, dim=1)
    print(f"余弦相似度:")
    print(f"  均值: {cos_sim.mean().item():.6f}")
    print(f"  方差: {cos_sim.var().item():.6f}")
    print(f"  标准差: {cos_sim.std().item():.6f}")
    print(f"  范围: [{cos_sim.min().item():.4f}, {cos_sim.max().item():.4f}]")

    # 欧氏距离
    euclidean_dist = torch.norm(emb_l - emb_orig, dim=1)
    print(f"\n欧氏距离:")
    print(f"  均值: {euclidean_dist.mean().item():.6f}")
    print(f"  标准差: {euclidean_dist.std().item():.6f}")

    # 方差对比
    var_l = emb_l.var(dim=0).mean().item()
    var_orig = emb_orig.var(dim=0).mean().item()
    print(f"\n方差 (按维度平均):")
    print(f"  训练后 (_l): {var_l:.6f}")
    print(f"  原始:        {var_orig:.6f}")
    print(f"  变化率:      {((var_l / var_orig) - 1) * 100:+.2f}%")

    # 整体方差
    overall_var_l = emb_l.var().item()
    overall_var_orig = emb_orig.var().item()
    print(f"\n整体方差:")
    print(f"  训练后 (_l): {overall_var_l:.6f}")
    print(f"  原始:        {overall_var_orig:.6f}")
    print(f"  变化率:      {((overall_var_l / overall_var_orig) - 1) * 100:+.2f}%")

    return {
        'cos_sim_mean': cos_sim.mean().item(),
        'var_l': var_l,
        'var_orig': var_orig,
        'overall_var_l': overall_var_l,
        'overall_var_orig': overall_var_orig,
    }

def main():
    parser = argparse.ArgumentParser(description='比较训练前后的embedding相似度和方差')
    parser.add_argument('--kg_l', default=DEFAULT_CONFIG['kg_l'], help='训练后的item embedding路径')
    parser.add_argument('--kg', default=DEFAULT_CONFIG['kg'], help='原始item embedding路径')
    parser.add_argument('--kg_user_l', default=DEFAULT_CONFIG['kg_user_l'], help='训练后的user embedding路径')
    parser.add_argument('--kg_user', default=DEFAULT_CONFIG['kg_user'], help='原始user embedding路径')
    args = parser.parse_args()

    print("=" * 70)
    print("Embedding 相似度与方差对比")
    print("=" * 70)

    # 加载文件
    print("\n加载embedding文件...")
    kg_l = load_embedding(args.kg_l)
    kg = load_embedding(args.kg)
    kg_user_l = load_embedding(args.kg_user_l)
    kg_user = load_embedding(args.kg_user)

    print(f"Item embedding: {kg_l.shape}")
    print(f"User embedding: {kg_user_l.shape}")

    # 比较
    item_stats = compare_embeddings(kg_l, kg, "Item Embeddings (kg_l vs kg)")
    user_stats = compare_embeddings(kg_user_l, kg_user, "User Embeddings (kg_user_l vs kg_user)")

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"{'指标':<25} {'Item':<15} {'User':<15}")
    print("-" * 55)
    print(f"{'余弦相似度':<25} {item_stats['cos_sim_mean']:<15.4f} {user_stats['cos_sim_mean']:<15.4f}")
    print(f"{'方差变化率':<25} {((item_stats['var_l']/item_stats['var_orig'])-1)*100:+.2f}%{'':<9} {((user_stats['var_l']/user_stats['var_orig'])-1)*100:+.2f}%")
    print(f"{'整体方差变化率':<25} {((item_stats['overall_var_l']/item_stats['overall_var_orig'])-1)*100:+.2f}%{'':<9} {((user_stats['overall_var_l']/user_stats['overall_var_orig'])-1)*100:+.2f}%")

if __name__ == '__main__':
    main()
