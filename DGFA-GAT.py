import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.autograd import Function
import pickle
os.environ["PYTHONHASHSEED"] = "270"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 需要
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"          # 方便排查
import warnings
warnings.filterwarnings("ignore", message="scatter_reduce_cuda does not have a deterministic implementation")
import random, numpy as np, torch
random.seed(270)
np.random.seed(270)
torch.manual_seed(270)
torch.cuda.manual_seed_all(270)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
#
#
# torch.set_num_threads(1)
# os.environ["OMP_NUM_THREADS"] = "1"  # 设置 OpenMP 线程数
# os.environ["MKL_NUM_THREADS"] = "1"  # 设置 MKL 线程数
# =========================

# =========================
# 固定随机种子
# =========================
def set_seed(seed: int = 142):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(270)


# =========================
# 超参（新增：伪标签自训练）

PSEUDO_START_EPOCH = 10      # 温启动：从第 10 个 epoch 开始做伪标签
PSEUDO_INTERVAL = 5          # 每隔 K 个 epoch 做一次伪标签
PSEUDO_W = 0.3               # 伪标签损失基准权重
PSEUDO_THRESH_EARLY = 0.9   # 前期置信度阈值
PSEUDO_THRESH_LATE = 0.8    # 后期置信度阈值（降低）
PSEUDO_LATE_RATIO = 0.6      # 进入“后期”的比例阈值：当 epoch/EPOCHS >= 0.6 时使用 0.75


# =========================
# GRL
# =========================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# =========================
# Focal Loss（已支持类别权重）
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=0.25, reduction='mean', class_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if class_weight is not None:
            self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))
        else:
            self.class_weight = None

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none', weight=self.class_weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =========================
# 自适应跳跃惩罚模块（稳定可学习权重：softmax 归一化）
# =========================
class AdaptiveJumpPenalty(nn.Module):
    def __init__(self):
        super(AdaptiveJumpPenalty, self).__init__()
        self.raw = nn.Parameter(torch.zeros(3, dtype=torch.float))  # 初始等权

    def forward(self, logits_src, logits_tgt, edge_index_src, edge_index_tgt,
                cluster_labels_src, cluster_labels_tgt, depth_src, depth_tgt):
        probs_src = F.softmax(logits_src, dim=1)
        i_src, j_src = edge_index_src
        pred_diff_src = torch.norm(probs_src[i_src] - probs_src[j_src], dim=1)
        depth_diff_src = torch.abs(depth_src[i_src] - depth_src[j_src])
        cluster_same_src = (cluster_labels_src[i_src] == cluster_labels_src[j_src])

        probs_tgt = F.softmax(logits_tgt, dim=1)
        i_tgt, j_tgt = edge_index_tgt
        pred_diff_tgt = torch.norm(probs_tgt[i_tgt] - probs_tgt[j_tgt], dim=1)
        depth_diff_tgt = torch.abs(depth_tgt[i_tgt] - depth_tgt[j_tgt])
        cluster_same_tgt = (cluster_labels_tgt[i_tgt] == cluster_labels_tgt[j_tgt])

        pred_diff = torch.cat([pred_diff_src, pred_diff_tgt])
        depth_diff = torch.cat([depth_diff_src, depth_diff_tgt])
        cluster_same = torch.cat([cluster_same_src, cluster_same_tgt])

        depth_diff_combined = torch.cat([depth_diff_src, depth_diff_tgt])
        boundary_threshold = torch.quantile(depth_diff_combined, 0.9)
        _ = (depth_diff > boundary_threshold) & (~cluster_same)  # 预留扩展

        device = pred_diff.device
        penalty_cluster = pred_diff[cluster_same].mean() if cluster_same.any() else torch.tensor(0.0, device=device)
        penalty_depth = pred_diff[depth_diff < 3.0].mean() if (depth_diff < 3.0).any() else torch.tensor(0.0, device=device)
        dynamic_threshold = torch.quantile(pred_diff, 0.9)
        mask_jump = pred_diff > dynamic_threshold
        penalty_jump = pred_diff[mask_jump].mean() if mask_jump.any() else torch.tensor(0.0, device=device)

        w = torch.softmax(self.raw, dim=0)
        total_penalty = w[0] * penalty_cluster + w[1] * penalty_depth + w[2] * penalty_jump
        return total_penalty

    def get_coefficients(self):
        with torch.no_grad():
            w = torch.softmax(self.raw, dim=0)
            return w[0].item(), w[1].item(), w[2].item()


# =========================
# 构建样本图（使用真实数据）
# =========================
def build_sample_graph(path, scaler=None, fit_scaler=False, depth_thresh=0.5):

    df = pd.read_csv(path).dropna()
    print(f"成功加载数据: {path}, 样本数: {len(df)}")

    required_cols = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies", "Depth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 {col} 在数据中不存在")

    feature_cols = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
    features = df[feature_cols].values
    labels = df["Facies"].values - 1
    depths = df["Depth"].values

    if fit_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print("拟合标准化器")
    else:
        assert scaler is not None, "scaler 为 None，请先在 fit_scaler=True 的数据上拟合"
        features = scaler.transform(features)

    edge_set = set()
    for i in range(len(depths) - 1):
        if abs(depths[i + 1] - depths[i]) <= depth_thresh:
            edge_set.add((i, i + 1))
            edge_set.add((i + 1, i))

    clustering = AffinityPropagation(random_state=270, damping=0.7).fit(features)
    cluster_labels = clustering.labels_
    cluster_tensor = torch.tensor(cluster_labels, dtype=torch.long)

    for cl in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cl)[0]
        if len(indices) < 2:
            continue
        cluster_features = features[indices]
        dists = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dist = np.linalg.norm(cluster_features[i] - cluster_features[j])
                dists.append(dist)
        if len(dists) == 0:
            continue
        threshold = np.percentile(dists, 25)
        for idx_i, i in enumerate(indices):
            for idx_j, j in enumerate(indices):
                if i < j and np.linalg.norm(features[i] - features[j]) < threshold:
                    edge_set.add((i, j))
                    edge_set.add((j, i))

    if len(edge_set) == 0:
        edge_set = {(i, i) for i in range(len(features))}

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    depth_tensor = torch.tensor(depths, dtype=torch.float)

    print(f"构建图完成: 节点数={len(x)}, 边数={edge_index.shape[1]}, 类别数={len(np.unique(labels))}")
    return Data(x=x, edge_index=edge_index, y=y), scaler, depth_tensor, cluster_tensor


# =========================
# 构建聚类图
# =========================
def build_cluster_graph(features, k_clusters=9, k_edges=20):
    N, D = features.shape
    kmeans = KMeans(n_clusters=k_clusters, random_state=270, n_init=20)
    cluster_ids = kmeans.fit_predict(features)
    Z = k_clusters

    Q = torch.zeros((N, Z))
    for i in range(N):
        Q[i, cluster_ids[i]] = 1

    cluster_sizes = Q.sum(dim=0, keepdim=True).T
    V = (Q.T @ torch.tensor(features, dtype=torch.float)) / (cluster_sizes + 1e-8)

    edge_list = []
    V_np = V.numpy()
    for i in range(Z):
        dists = np.linalg.norm(V_np[i] - V_np, axis=1)
        nearest = np.argsort(dists)[1:k_edges + 1]
        for j in nearest:
            edge_list.append((i, j))
            edge_list.append((j, i))
    if len(edge_list) == 0:
        edge_list = [(i, i) for i in range(Z)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=V, edge_index=edge_index), Q


# =========================
# 双图融合 GAT 模型
# =========================
class GraphBasedDA_GAT(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.gat1_s = GATConv(input_dim, 256, heads=32)
        self.gat2_s = GATConv(256 * 32, 128, heads=1)
        self.gat1_c = GATConv(input_dim, 256, heads=32)
        self.gat2_c = GATConv(256 * 32, 128, heads=1)

        self.attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Linear(128, 9)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.jump_penalty = AdaptiveJumpPenalty()

    def forward(self, sample_data, cluster_data, Q, alpha=1.0):
        xs = F.relu(self.gat1_s(sample_data.x, sample_data.edge_index))
        xs = F.relu(self.gat2_s(xs, sample_data.edge_index))

        xc = F.relu(self.gat1_c(cluster_data.x, cluster_data.edge_index))
        xc = F.relu(self.gat2_c(xc, cluster_data.edge_index))

        xc_decoded = Q @ xc
        fusion = torch.cat([xs, xc_decoded], dim=1)
        weights = self.attn(fusion)
        fused = weights[:, 0:1] * xs + weights[:, 1:2] * xc_decoded

        class_output = self.classifier(fused)
        reversed_feat = GradientReversalFunction.apply(fused, alpha)
        domain_output = self.domain_discriminator(reversed_feat)

        return class_output, domain_output, fused


# =========================
# 邻接矩阵准确率
# =========================
def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0, nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / conf.sum()


# =========================
# GAT 引导的随机游走过采样
# =========================
def gat_random_walk_oversample(data: Data, depth_tensor, cluster_tensor,
                               model: GraphBasedDA_GAT,
                               device: torch.device,
                               walk_len: int = 9,
                               synth_per_class_to_max: bool = True,
                               per_node_synth: int = 1):
    model.eval()
    x = data.x.clone()
    y = data.y.clone()
    edge_index = data.edge_index.clone()
    depth = depth_tensor.clone()
    cluster = cluster_tensor.clone()

    with torch.no_grad():
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        attn = torch.ones(len(src), device=device) * 0.5

    out_neigh = defaultdict(list)
    out_w = defaultdict(list)
    for s, d, w in zip(src, dst, attn.tolist()):
        out_neigh[s].append(d)
        out_w[s].append(max(float(w), 1e-9))

    out_prob = {}
    for s, ws in out_w.items():
        arr = np.asarray(ws, dtype=float)
        ssum = arr.sum()
        if ssum <= 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / ssum
        out_prob[s] = arr

    binc = torch.bincount(y, minlength=int(y.max().item()) + 1)
    max_count = int(binc.max().item())
    num_classes = int(y.max().item()) + 1

    new_feats = []
    new_labels = []
    new_depths = []
    new_clusters = []
    new_edges = set([tuple(e) for e in edge_index.t().tolist()])

    for c in range(num_classes):
        idx_c = (y == c).nonzero(as_tuple=False).view(-1).tolist()
        if len(idx_c) == 0:
            continue
        if synth_per_class_to_max:
            need = max(0, max_count - len(idx_c))
        else:
            need = len(idx_c) * per_node_synth
        if need == 0:
            continue

        made = 0
        ptr = 0
        while made < need:
            start = idx_c[ptr % len(idx_c)]
            ptr += 1

            cur = start
            feat = x[start].clone()
            cur_depth = depth[start].clone()
            cur_cluster = cluster[start].clone()

            for _ in range(walk_len):
                if cur not in out_neigh or len(out_neigh[cur]) == 0:
                    break
                neighs = out_neigh[cur]
                probs = out_prob[cur]
                nxt = int(np.random.choice(neighs, p=probs))
                lam = float(np.random.rand())
                feat = (1 - lam) * feat + lam * x[nxt]
                cur_depth = (1 - lam) * cur_depth + lam * depth[nxt]
                if random.random() < 0.5:
                    cur_cluster = cluster[start]
                else:
                    cur_cluster = cluster[nxt]
                cur = nxt

            new_feats.append(feat.unsqueeze(0))
            new_labels.append(c)
            new_depths.append(cur_depth.unsqueeze(0))
            new_clusters.append(cur_cluster.unsqueeze(0))

            new_id = int(x.size(0) + len(new_feats) - 1)
            new_edges.add((new_id, start))
            new_edges.add((start, new_id))
            if start != cur:
                new_edges.add((new_id, cur))
                new_edges.add((cur, new_id))
            else:
                if start in out_neigh and len(out_neigh[start]) > 0:
                    j = out_neigh[start][0]
                    new_edges.add((new_id, j))
                    new_edges.add((j, new_id))

            made += 1

    if len(new_feats) > 0:
        x = torch.cat([x, torch.cat(new_feats, dim=0)], dim=0)
        y = torch.cat([y, torch.tensor(new_labels, dtype=torch.long)], dim=0)
        depth = torch.cat([depth, torch.cat(new_depths, dim=0)], dim=0)
        cluster = torch.cat([cluster, torch.cat(new_clusters, dim=0)], dim=0)
        edge_index = torch.tensor(sorted(list(new_edges)), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y), depth, cluster


# =========================
# 训练主流程（伪标签：最小侵入式稳健版 + 方案B）
# =========================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("构建训练数据...")
    train_data, scaler, depth_train, cluster_train = build_sample_graph("dataset/WA.csv",
                                                                        fit_scaler=True)
    print("构建测试数据...")
    test_data, _, depth_test, cluster_test = build_sample_graph("dataset/WB.csv", scaler=scaler)

    model = GraphBasedDA_GAT().to(device)

    print("进行过采样...")
    train_data, depth_train, cluster_train = gat_random_walk_oversample(
        data=train_data,
        depth_tensor=depth_train,
        cluster_tensor=cluster_train,
        model=model,
        device=device,
        walk_len=1,
        synth_per_class_to_max=True
    )
    print(f"过采样后训练数据大小: {train_data.x.shape[0]}")

    print("构建聚类图...")
    cluster_data_train, Q_train = build_cluster_graph(train_data.x.cpu().numpy())
    cluster_data_test, Q_test = build_cluster_graph(test_data.x.cpu().numpy())

    cluster_data_train, cluster_data_test = cluster_data_train.to(device), cluster_data_test.to(device)
    Q_train, Q_test = Q_train.to(device), Q_test.to(device)
    train_data, test_data = train_data.to(device), test_data.to(device)
    depth_train, depth_test = depth_train.to(device), depth_test.to(device)
    cluster_train, cluster_test = cluster_train.to(device), cluster_test.to(device)

    # === 类别加权（来自源域/训练集的标签频次；逆频率并做均值归一） ===
    num_classes = 9
    binc = torch.bincount(train_data.y, minlength=num_classes).float()
    # 防止除0
    binc = torch.clamp(binc, min=1.0)
    inv_freq = 1.0 / binc
    class_weights = (inv_freq / inv_freq.mean()).to(device)

    # FocalLoss（带权）
    criterion_cls = FocalLoss(gamma=3, alpha=0.25, class_weight=class_weights).to(device)

    domain_ce = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 80
    best_f1 = -1.0
    best_metrics = {"epoch": 0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    os.makedirs("pseudo_out", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # DANN GRL 调度
        p = epoch / EPOCHS
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        stage = "Stage 1"

        # 前向（拿到 fused_src / fused_tgt 用于伪标签）
        out_src, dom_src, _ = model(train_data, cluster_data_train, Q_train, alpha)
        out_tgt, dom_tgt, _ = model(test_data, cluster_data_test, Q_test, alpha)

        # 分类损失（源域有标签）—— 已加权 FocalLoss
        loss_cls = criterion_cls(out_src, train_data.y)

        # 跳跃惩罚
        loss_jump = model.jump_penalty(
            out_src, out_tgt,
            train_data.edge_index, test_data.edge_index,
            cluster_train, cluster_test,
            depth_train, depth_test
        )

        # 域适应损失
        domain_labels = torch.cat([
            torch.zeros(len(train_data.x), dtype=torch.long, device=device),
            torch.ones(len(test_data.x), dtype=torch.long, device=device)
        ], dim=0)
        domain_outputs = torch.cat([dom_src, dom_tgt], dim=0)
        loss_dom = domain_ce(domain_outputs, domain_labels).mean()

        # ======= 伪标签自训练（方案B：80%分位兜底 + 后期降权退火） =======
        loss_pseudo = torch.tensor(0.0, device=device)
        num_pseudo = 0
        use_pseudo = (epoch >= PSEUDO_START_EPOCH) and ((epoch - PSEUDO_START_EPOCH) % PSEUDO_INTERVAL == 0)
        if use_pseudo:
            with torch.no_grad():
                probs_tgt = F.softmax(out_tgt, dim=1)
                max_prob, pseudo_y = probs_tgt.max(dim=1)
                # 动态阈值
                th = PSEUDO_THRESH_LATE if (epoch / EPOCHS) >= PSEUDO_LATE_RATIO else PSEUDO_THRESH_EARLY
                # ---- 80%分位兜底（阈值取 max(th, p80)）
                p80 = torch.quantile(max_prob, 0.80).item() if max_prob.numel() > 0 else th
                th = max(th, p80)

                mask = max_prob >= th
                idx_all = torch.nonzero(mask, as_tuple=False).view(-1)

                # ---- Top-K 全局 + 每类 Top-K ----
                MAX_PSEUDO_PER_EPOCH = 64
                MAX_PSEUDO_PER_CLASS = 10

                if idx_all.numel() > 0:
                    conf_all = max_prob[idx_all]
                    sort_conf, order = torch.sort(conf_all, descending=True)
                    idx_all = idx_all[order]
                    # 全局截断
                    idx_all = idx_all[:MAX_PSEUDO_PER_EPOCH]

                    # 按预测类分桶取每类 top-k
                    sel = []
                    counts = defaultdict(int)
                    for ii in idx_all.tolist():
                        c = int(pseudo_y[ii].item())
                        if counts[c] < MAX_PSEUDO_PER_CLASS:
                            sel.append(ii)
                            counts[c] += 1
                    if len(sel) > 0:
                        idx = torch.tensor(sel, device=device, dtype=torch.long)
                    else:
                        idx = torch.tensor([], device=device, dtype=torch.long)
                else:
                    idx = idx_all  # 空

                num_pseudo = int(idx.numel())

            if num_pseudo > 0:
                # 置信度加权的伪标签损失（样本内权重）
                ce_each = F.cross_entropy(out_tgt[idx], pseudo_y[idx], reduction='none')
                conf = max_prob[idx]
                w_sample = ((conf - th) / (1.0 - th)).clamp(min=0.0, max=1.0)  # 线性映射到 [0,1]
                # 数量越多，整体权重越小（保护主任务）
                effective_w = PSEUDO_W * (64 / num_pseudo) ** 0.5
                # 后期降权退火（p = epoch / EPOCHS）
                anneal = (1.0 - p) ** 0.5
                effective_w = effective_w * anneal

                loss_pseudo = (ce_each * w_sample).mean() * effective_w

        # 总损失
        total_loss = loss_cls + loss_dom + 0.2* loss_jump + 0.4*loss_pseudo
        # total_loss = loss_cls + loss_dom + loss_pseudo
        total_loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            out_eval, _, _ = model(test_data, cluster_data_test, Q_test, alpha=1.0)
            out_src_eval, _, _ = model(train_data, cluster_data_train, Q_train, alpha=1.0)

            pred_src = out_src_eval.argmax(dim=1)
            acc_src = (pred_src == train_data.y).float().mean().item()

            pred_tgt = out_eval.argmax(dim=1).cpu()
            true_tgt = test_data.y.cpu()
            acc_tgt = (pred_tgt == true_tgt).float().mean().item()

            precision = precision_score(true_tgt, pred_tgt, average='macro', zero_division=0)
            recall = recall_score(true_tgt, pred_tgt, average='macro', zero_division=0)
            f1 = f1_score(true_tgt, pred_tgt, average='macro', zero_division=0)

            a, b, c = model.jump_penalty.get_coefficients()

        print(f"Epoch {epoch:04d} | {stage} | alpha: {alpha:.3f} | "
              f"Loss: {total_loss:.4f} | "
              f"PseudoUsed: {use_pseudo} | PseudoN: {num_pseudo} | "
              f"Src Acc: {acc_src:.4f} | Tgt Acc: {acc_tgt:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
              f"Weights: a={a:.3f}, b={b:.3f}, c={c:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {"epoch": epoch, "acc": acc_tgt, "precision": precision, "recall": recall, "f1": f1}
            torch.save(model.state_dict(), "pseudo_out/DGFA-GAT_a_b_best_model.pth")
            np.savetxt("pseudo_out/DGFA-GAT_a_b.txt", pred_tgt.numpy(), fmt='%d')
            with open("pseudo_out/DGFA-GAT_a_b.pkl", "wb") as f:
                pickle.dump(confusion_matrix(true_tgt, pred_tgt), f)

    # === 训练结束：打印最佳指标 + 邻接矩阵准确率 ===
    try:
        with open("pseudo_out/DGFA-GAT_a_b.pkl", "rb") as f:
            conf_matrix = pickle.load(f)



        print("\n================ 最佳模型性能 ================\n")
        print(f"最佳 Epoch: {best_metrics['epoch']}")
        print(f"准确率 Accuracy: {best_metrics['acc']:.4f}")
        print(f"精准率 Precision: {best_metrics['precision']:.4f}")
        print(f"召回率 Recall: {best_metrics['recall']:.4f}")
        print(f"F1-score: {best_metrics['f1']:.4f}")
        print("混淆矩阵：")
        print(conf_matrix)

        a, b, c = model.jump_penalty.get_coefficients()
        print(f"最终学习到的权重: a={a:.4f}, b={b:.4f}, c={c:.4f}")

    except FileNotFoundError:
        print("训练过程中未保存最佳模型，可能所有epoch的F1都为0")

    print("模型已保存为 DGFA_GAT_best_model.pth")



if __name__ == '__main__':
    train_model()

