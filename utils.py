"""
Utility Functions
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def set_seed(seed: int = 270):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def gat_random_walk_oversample(data: Data, depth_tensor, cluster_tensor,
                               model, device: torch.device,
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