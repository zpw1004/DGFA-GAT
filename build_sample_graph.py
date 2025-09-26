"""
Sample Graph Construction Function
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

def build_sample_graph(path, scaler=None, fit_scaler=False, depth_thresh=0.5):
    df = pd.read_csv(path).dropna()
    print(f"Successfully loaded data: {path}, number of samples: {len(df)}")

    required_cols = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies", "Depth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing in the data")

    feature_cols = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
    features = df[feature_cols].values
    labels = df["Facies"].values - 1
    depths = df["Depth"].values

    if fit_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print("Fitted the scaler")
    else:
        assert scaler is not None, "Scaler is None, please fit the scaler with fit_scaler=True"
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

    print(f"Graph construction complete: Number of nodes={len(x)}, Number of edges={edge_index.shape[1]}, Number of classes={len(np.unique(labels))}")
    return Data(x=x, edge_index=edge_index, y=y), scaler, depth_tensor, cluster_tensor
