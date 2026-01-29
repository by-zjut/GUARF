# nettack_pytorch.py
# 完整实现：Nettack 图结构攻击 + PyTorch GCN（修复梯度、效率、评估问题）

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import argparse
import warnings
import os

warnings.filterwarnings("ignore")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()


# ========================
# 1. 数据加载与预处理
# ========================

def load_pt(file_name):
    """
    Load graph data from .pt file.
    Expected format:
        data = (dict_with_x_edge_index_y, None, DataClassType)
    """
    data = torch.load(file_name)

    if isinstance(data, tuple) and len(data) >= 3:
        print(f"Loaded data structure detected: tuple with {len(data)} elements.")
        data_dict = data[0]  # 第一个元素是包含 x, edge_index, y 的字典
        if not isinstance(data_dict, dict):
            raise ValueError("First element of the tuple should be a dict.")
        if 'x' not in data_dict or 'edge_index' not in data_dict or 'y' not in data_dict:
            raise ValueError("Dictionary must contain 'x', 'edge_index', and 'y' keys.")
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    # 提取张量
    x = data_dict['x']
    edge_index = data_dict['edge_index']
    y = data_dict['y']

    # 转为 numpy
    x = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    edge_index = edge_index.numpy() if hasattr(edge_index, 'numpy') else np.array(edge_index)
    y = y.numpy() if hasattr(y, 'numpy') else np.array(y)

    # 如果 y 是 one-hot 编码（二维），转换为类别标签
    if y.ndim == 2:
        y = y.argmax(axis=1)
    elif y.ndim > 2:
        raise ValueError(f"Labels y have unsupported shape: {y.shape}")

    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]

    print(f"Loaded: {num_nodes} nodes, {num_edges} edges, {len(np.unique(y))} classes")

    # 构建邻接矩阵（无向图）
    adj = sp.csr_matrix((np.ones(num_edges), (edge_index[0], edge_index[1])),
                        shape=(num_nodes, num_nodes))
    adj = adj + adj.T  # 转为无向图
    adj = adj > 0  # 去重（避免多条边）
    adj = adj.astype(np.float32)

    features = sp.csr_matrix(x)
    labels = y

    return adj, features, labels

def preprocess_graph(adj):
    """GCN-style symmetric normalization: Â = D^{-1/2} (A + I) D^{-1/2}"""
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    degree_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized


def sparse_to_torch_sparse(matrix):
    """Convert scipy sparse matrix to torch sparse tensor."""
    if not isinstance(matrix, sp.coo_matrix):
        matrix = matrix.tocoo()
    indices = torch.LongTensor(np.vstack((matrix.row, matrix.col)))
    values = torch.FloatTensor(matrix.data)
    shape = torch.Size(matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


# ========================
# 2. PyTorch GCN 模型
# ========================

class GraphConvolution(nn.Module):
    """Simple GCN layer: X' = \sigma(Â @ X @ W + b)"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# ========================
# 3. Nettack 攻击核心（优化版）
# ========================

class Nettack:
    def __init__(self, model, adj, features, labels, idx_attack,
                 n_perturbations=5, lambda_=0.5, device='cpu'):
        """
        Nettack: Structure attack on GCN via edge perturbation.
        :param model: Trained GCN model
        :param adj: Original adjacency matrix (scipy csr)
        :param features: Node features (scipy csr)
        :param labels: Ground truth labels
        :param idx_attack: Target node index
        :param n_perturbations: Number of edge flips
        :param lambda_: Weight for degree regularization (0~1)
        :param device: 'cuda' or 'cpu'
        """
        self.model = model
        self.adj = adj.tocsr()
        self.features = features
        self.labels = labels
        self.idx_attack = idx_attack
        self.n_perturbations = n_perturbations
        self.lambda_ = lambda_
        self.device = device

        self.modified_adj = self.adj.copy()
        self.ori_degree = np.array(self.adj.sum(1)).flatten()

        # 转为 GPU 张量（仅一次）
        self.feat_tensor = torch.FloatTensor(self.features.todense()).to(device)
        self.labels_tensor = torch.LongTensor(self.labels).to(device)

    def get_candidates(self):
        """
        Generate candidate edges: (add: non-neighbors with same class), (remove: neighbors)
        Apply degree constraint: deg(v) >= 2
        """
        N = self.adj.shape[0]
        u = self.idx_attack
        neighbors = self.modified_adj[u].nonzero()[1]
        non_neighbors = np.setdiff1d(np.arange(N), neighbors)

        # Only consider same-class nodes
        same_class = np.where(self.labels == self.labels[u])[0]
        can_add = np.intersect1d(non_neighbors, same_class)
        can_remove = neighbors

        # Filter by degree constraint (after flip, degree >= 2)
        valid_add = []
        for v in can_add:
            if v == u: continue
            if self.modified_adj[v].sum() + 1 >= 2:  # after adding edge
                valid_add.append((u, v, 1))

        valid_remove = []
        for v in can_remove:
            if v == u: continue
            if self.modified_adj[v].sum() - 1 >= 1:  # after removing edge
                valid_remove.append((u, v, -1))

        candidates = valid_add + valid_remove
        return candidates

    def estimate_perturbation_effect(self, u, v):
        """
        Estimate effect of flipping edge (u,v) on loss of target node.
        Uses finite difference.
        """
        self.model.eval()
        with torch.no_grad():
            # Clean prediction
            adj_clean = sparse_to_torch_sparse(self.modified_adj).to(self.device)
            logits_clean = self.model(self.feat_tensor, adj_clean)
            loss_clean = F.nll_loss(logits_clean[u:u+1], self.labels_tensor[u:u+1]).item()

            # Flip edge
            self.modified_adj[u, v] = 1 - self.modified_adj[u, v]
            self.modified_adj[v, u] = 1 - self.modified_adj[u, v]

            adj_perturb = sparse_to_torch_sparse(self.modified_adj).to(self.device)
            logits_perturb = self.model(self.feat_tensor, adj_perturb)
            loss_perturb = F.nll_loss(logits_perturb[u:u+1], self.labels_tensor[u:u+1]).item()

            # Restore
            self.modified_adj[u, v] = 1 - self.modified_adj[u, v]
            self.modified_adj[v, u] = 1 - self.modified_adj[u, v]

            # Loss increase = good for attacker
            loss_diff = loss_perturb - loss_clean

            # Degree regularization: penalize high-degree nodes
            deg_u_after = self.modified_adj[u].sum() + (1 if self.modified_adj[u,v]==0 else -1)
            deg_v_after = self.modified_adj[v].sum() + (1 if self.modified_adj[u,v]==0 else -1)
            deg_penalty = np.log(deg_u_after) + np.log(deg_v_after) - np.log(self.ori_degree[u]) - np.log(self.ori_degree[v])

            score = loss_diff - self.lambda_ * deg_penalty
            return score

    def attack(self):
        print(f"Attacking node {self.idx_attack}, true label: {self.labels[self.idx_attack]}")
        for step in range(self.n_perturbations):
            candidates = self.get_candidates()
            if len(candidates) == 0:
                print(f"Step {step+1}: No candidates left.")
                break

            best_score = -np.inf
            best_edge = None

            for u, v, action in candidates:
                score = self.estimate_perturbation_effect(u, v)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)

            if best_edge is None:
                continue

            u, v = best_edge
            # Flip edge
            current = self.modified_adj[u, v]
            self.modified_adj[u, v] = 1 - current
            self.modified_adj[v, u] = 1 - current

            # Check misclassification
            adj_tensor = sparse_to_torch_sparse(self.modified_adj).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(self.feat_tensor, adj_tensor)[self.idx_attack].argmax().item()
                true = self.labels[self.idx_attack]
                if pred != true:
                    print(f"Step {step+1}: Node {self.idx_attack} successfully attacked! "
                          f"(True: {true}, Pred: {pred})")
                    break

        print(f"Attack finished. Final degree: {self.modified_adj[self.idx_attack].sum()}")

    def get_modified_adj(self):
        """Return the perturbed adjacency matrix."""
        return self.modified_adj.copy()


# ========================
# 4. 训练与评估
# ========================

def train_gcn(model, adj, features, labels, idx_train, idx_val,
              epochs=200, lr=0.01, weight_decay=5e-4, device='cpu'):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    adj_tensor = sparse_to_torch_sparse(adj).to(device)
    feat_tensor = torch.FloatTensor(features.todense()).to(device)
    labels_tensor = torch.LongTensor(labels).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(feat_tensor, adj_tensor)
        loss = F.nll_loss(output[idx_train], labels_tensor[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = output[idx_val].max(1)[1]
                acc = pred.eq(labels_tensor[idx_val]).sum().item() / len(idx_val)
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}')
    model.eval()


def evaluate(model, adj, features, labels, idx_test, device='cpu'):
    model.eval()
    adj_tensor = sparse_to_torch_sparse(adj).to(device)
    feat_tensor = torch.FloatTensor(features.todense()).to(device)
    labels_tensor = torch.LongTensor(labels).to(device)

    with torch.no_grad():
        output = model(feat_tensor, adj_tensor)
        pred = output[idx_test].max(1)[1]
        labels_test = labels_tensor[idx_test]
        acc = pred.eq(labels_test).sum().item() / len(idx_test)
        f1_mi = f1_score(labels_test.cpu(), pred.cpu(), average='micro')
        f1_ma = f1_score(labels_test.cpu(), pred.cpu(), average='macro')
    return acc, f1_mi, f1_ma


# ========================
# 5. 主函数（批量攻击测试）
# ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed',
                        choices=['PubMed'])
    parser.add_argument('--data_path', type=str,
                        default='/home/Newdisk2/baiyang/GLM_Eva/model/TAPE/dataset/PubMed/processed/data.pt')
    parser.add_argument('--n_attack', type=int, default=10, help="Number of target nodes to attack")
    parser.add_argument('--n_perturb', type=int, default=5, help="Number of edge perturbations per node")
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading data...")
    adj, features, labels = load_pt(args.data_path)
    adj = adj + adj.T
    adj[adj > 1] = 1
    adj = preprocess_graph(adj)
    features = sp.csr_matrix(features)

    # 2. 划分数据集
    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(idx, test_size=0.3, stratify=labels, random_state=42)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.2, stratify=labels[idx_train], random_state=42)

    # 3. 初始化并训练 GCN
    n_class = labels.max() + 1
    model = GCN(in_features=features.shape[1], hidden_features=args.hidden,
                out_features=n_class, dropout=args.dropout).to(device)
    print(f"Model: GCN({features.shape[1]} -> {args.hidden} -> {n_class})")

    print("Training clean model...")
    train_gcn(model, adj, features, labels, idx_train, idx_val, epochs=200, device=device)

    # 4. 评估干净模型
    acc_clean, f1_mi_clean, f1_ma_clean = evaluate(model, adj, features, labels, idx_test, device)
    print(f"\n[Clean Model] Acc: {acc_clean:.4f}, F1-micro: {f1_mi_clean:.4f}, F1-macro: {f1_ma_clean:.4f}")

    # 5. 批量攻击测试
    np.random.seed(42)
    attack_nodes = np.random.choice(idx_test, size=min(args.n_attack, len(idx_test)), replace=False)
    success_count = 0
    modified_adjs = []

    for node_idx in attack_nodes:
        print(f"\n--- Attacking Node {node_idx} ---")
        attacker = Nettack(model, adj, features, labels, node_idx,
                           n_perturbations=args.n_perturb, device=device)
        attacker.attack()

        # Check if attack succeeded
        modified_adj = attacker.get_modified_adj()
        modified_adj = preprocess_graph(modified_adj)  # 重新归一化
        acc_after, _, _ = evaluate(model, modified_adj, features, labels, [node_idx], device)
        pred_after = model(torch.FloatTensor(features.todense()).to(device),
                           sparse_to_torch_sparse(modified_adj).to(device))[node_idx].argmax().item()
        true_label = labels[node_idx]

        if pred_after != true_label:
            success_count += 1
            print(f"✅ Attack SUCCESS on node {node_idx}")
        else:
            print(f"❌ Attack FAILED on node {node_idx}")

        modified_adjs.append(modified_adj)

    # 6. 汇总结果
    asr = success_count / len(attack_nodes)
    print(f"\n=== Final Results ===")
    print(f"Clean Accuracy: {acc_clean:.4f}")
    print(f"Attack Success Rate (ASR): {asr:.4f} ({success_count}/{len(attack_nodes)})")

    # 可选：保存扰动图
    # torch.save(modified_adjs, "perturbed_adjs.pt")


if __name__ == "__main__":
    main()