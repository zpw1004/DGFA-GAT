"""
Clustering Graph Construction Function
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans

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