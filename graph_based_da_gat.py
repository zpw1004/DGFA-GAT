"""
DGFA-GAT: Dual-Graph Fusion Adaptive Graph Attention Network

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.autograd import Function



class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


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


class AdaptiveJumpPenalty(nn.Module):
    def __init__(self):
        super(AdaptiveJumpPenalty, self).__init__()
        self.raw = nn.Parameter(torch.zeros(3, dtype=torch.float))

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


class GraphBasedDA_GAT(nn.Module):
    def __init__(self, input_dim=7, output_dim=9):
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

        self.classifier = nn.Linear(128,output_dim)
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