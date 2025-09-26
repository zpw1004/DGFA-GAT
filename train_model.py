"""
Main training flow file - using command line arguments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from collections import defaultdict

# Import configurations
from config import (
    DATA_CONFIG, TRAIN_CONFIG, PATH_CONFIG, SEED_CONFIG, LOG_CONFIG, MODEL_CONFIG, GRAPH_CONFIG
)
from graph_based_da_gat import GraphBasedDA_GAT, FocalLoss
from build_sample_graph import build_sample_graph
from build_cluster_graph import build_cluster_graph
from utils import set_seed, gat_random_walk_oversample
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def print_configuration():
    """Print current configuration"""
    print("=" * 60)
    print("DGFA-GAT Training Configuration")
    print("=" * 60)
    print(f"Device: {TRAIN_CONFIG['device']}")
    print(f"Random Seed: {SEED_CONFIG['seed']}")
    print(f"Training Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"Source Domain Data: {DATA_CONFIG['source_path']}")
    print(f"Target Domain Data: {DATA_CONFIG['target_path']}")
    print(f"Output Directory: {PATH_CONFIG['output_dir']}")
    print(f"Input Dimension: {MODEL_CONFIG['input_dim']}")
    print(f"Output Dimension: {MODEL_CONFIG['output_dim']}")
    print(f"Pseudo-label Start Epoch: {TRAIN_CONFIG['pseudo_label']['start_epoch']}")
    print("=" * 60)


def train_model():
    set_seed(SEED_CONFIG['seed'])

    print_configuration()

    device = torch.device(TRAIN_CONFIG['device'])

    # Build data
    print("Building training data...")
    train_data, scaler, depth_train, cluster_train = build_sample_graph(
        DATA_CONFIG['source_path'],
        fit_scaler=True,
        depth_thresh=DATA_CONFIG['depth_threshold']
    )
    print("Building test data...")
    test_data, _, depth_test, cluster_test = build_sample_graph(
        DATA_CONFIG['target_path'],
        scaler=scaler,
        depth_thresh=DATA_CONFIG['depth_threshold']
    )

    model = GraphBasedDA_GAT(input_dim=MODEL_CONFIG['input_dim']).to(device)

    print("Performing oversampling...")
    oversample_cfg = TRAIN_CONFIG['oversampling']
    train_data, depth_train, cluster_train = gat_random_walk_oversample(
        data=train_data,
        depth_tensor=depth_train,
        cluster_tensor=cluster_train,
        model=model,
        device=device,
        walk_len=oversample_cfg['walk_length'],
        synth_per_class_to_max=oversample_cfg['synth_per_class_to_max'],
        per_node_synth=oversample_cfg['per_node_synth']
    )
    print(f"Oversampled training data size: {train_data.x.shape[0]}")

    print("Building clustering graph...")
    cluster_graph_cfg = GRAPH_CONFIG['cluster_graph']
    cluster_data_train, Q_train = build_cluster_graph(
        train_data.x.cpu().numpy(),
        k_clusters=cluster_graph_cfg['n_clusters'],
        k_edges=cluster_graph_cfg['n_edges']
    )
    cluster_data_test, Q_test = build_cluster_graph(
        test_data.x.cpu().numpy(),
        k_clusters=cluster_graph_cfg['n_clusters'],
        k_edges=cluster_graph_cfg['n_edges']
    )

    cluster_data_train, cluster_data_test = cluster_data_train.to(device), cluster_data_test.to(device)
    Q_train, Q_test = Q_train.to(device), Q_test.to(device)
    train_data, test_data = train_data.to(device), test_data.to(device)
    depth_train, depth_test = depth_train.to(device), depth_test.to(device)
    cluster_train, cluster_test = cluster_train.to(device), cluster_test.to(device)

    num_classes = MODEL_CONFIG['output_dim']
    binc = torch.bincount(train_data.y, minlength=num_classes).float()
    binc = torch.clamp(binc, min=1.0)
    inv_freq = 1.0 / binc
    class_weights = (inv_freq / inv_freq.mean()).to(device)

    focal_cfg = TRAIN_CONFIG['loss']['focal_loss']
    criterion_cls = FocalLoss(
        gamma=focal_cfg['gamma'],
        alpha=focal_cfg['alpha'],
        class_weight=class_weights
    ).to(device)

    domain_ce = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])

    # Training parameters
    epochs = TRAIN_CONFIG['epochs']
    pseudo_cfg = TRAIN_CONFIG['pseudo_label']
    loss_weights = TRAIN_CONFIG['loss']
    best_f1 = -1.0
    best_metrics = {"epoch": 0, "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    os.makedirs(PATH_CONFIG['output_dir'], exist_ok=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        p = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        out_src, dom_src, _ = model(train_data, cluster_data_train, Q_train, alpha)
        out_tgt, dom_tgt, _ = model(test_data, cluster_data_test, Q_test, alpha)

        loss_cls = criterion_cls(out_src, train_data.y)
        loss_jump = model.jump_penalty(
            out_src, out_tgt, train_data.edge_index, test_data.edge_index,
            cluster_train, cluster_test, depth_train, depth_test
        )

        domain_labels = torch.cat([
            torch.zeros(len(train_data.x), dtype=torch.long, device=device),
            torch.ones(len(test_data.x), dtype=torch.long, device=device)
        ], dim=0)
        domain_outputs = torch.cat([dom_src, dom_tgt], dim=0)
        loss_dom = domain_ce(domain_outputs, domain_labels).mean()

        loss_pseudo = torch.tensor(0.0, device=device)
        num_pseudo = 0
        use_pseudo = (epoch >= pseudo_cfg['start_epoch']) and \
                     ((epoch - pseudo_cfg['start_epoch']) % pseudo_cfg['interval'] == 0)

        if use_pseudo:
            with torch.no_grad():
                probs_tgt = F.softmax(out_tgt, dim=1)
                max_prob, pseudo_y = probs_tgt.max(dim=1)

                th = pseudo_cfg['threshold_late'] if (epoch / epochs) >= pseudo_cfg['late_ratio'] else pseudo_cfg[
                    'threshold_early']
                p80 = torch.quantile(max_prob, 0.80).item() if max_prob.numel() > 0 else th
                th = max(th, p80)

                mask = max_prob >= th
                idx_all = torch.nonzero(mask, as_tuple=False).view(-1)

                if idx_all.numel() > 0:
                    conf_all = max_prob[idx_all]
                    sort_conf, order = torch.sort(conf_all, descending=True)
                    idx_all = idx_all[order]
                    idx_all = idx_all[:pseudo_cfg['max_per_epoch']]

                    sel = []
                    counts = defaultdict(int)
                    for ii in idx_all.tolist():
                        c = int(pseudo_y[ii].item())
                        if counts[c] < pseudo_cfg['max_per_class']:
                            sel.append(ii)
                            counts[c] += 1
                    if len(sel) > 0:
                        idx = torch.tensor(sel, device=device, dtype=torch.long)
                    else:
                        idx = torch.tensor([], device=device, dtype=torch.long)
                else:
                    idx = idx_all

                num_pseudo = int(idx.numel())

            if num_pseudo > 0:
                ce_each = F.cross_entropy(out_tgt[idx], pseudo_y[idx], reduction='none')
                conf = max_prob[idx]
                w_sample = ((conf - th) / (1.0 - th)).clamp(min=0.0, max=1.0)
                effective_w = pseudo_cfg['base_weight'] * (pseudo_cfg['max_per_epoch'] / num_pseudo) ** 0.5
                anneal = (1.0 - p) ** 0.5
                effective_w = effective_w * anneal
                loss_pseudo = (ce_each * w_sample).mean() * effective_w

        total_loss = (loss_cls + loss_dom +
                      loss_weights['jump_penalty_weight'] * loss_jump +
                      loss_weights['pseudo_weight'] * loss_pseudo)
        total_loss.backward()
        optimizer.step()


        if epoch % LOG_CONFIG['print_freq'] == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                # 目标域预测
                out_eval, _, _ = model(test_data, cluster_data_test, Q_test, alpha=1.0)
                pred_tgt = out_eval.argmax(dim=1).cpu()
                true_tgt = test_data.y.cpu()

                # 源域预测（新增）
                out_src_eval, _, _ = model(train_data, cluster_data_train, Q_train, alpha=1.0)
                pred_src = out_src_eval.argmax(dim=1).cpu()
                true_src = train_data.y.cpu()


                acc_tgt = (pred_tgt == true_tgt).float().mean().item()
                acc_src = (pred_src == true_src).float().mean().item()
                precision = precision_score(true_tgt, pred_tgt, average='macro', zero_division=0)
                recall = recall_score(true_tgt, pred_tgt, average='macro', zero_division=0)
                f1 = f1_score(true_tgt, pred_tgt, average='macro', zero_division=0)


                w1, w2, w3 = model.jump_penalty.get_coefficients()


            stage = "Stage 2" if use_pseudo and num_pseudo > 0 else "Stage 1"


            print(f"Epoch {epoch:04d} | {stage} | alpha: {alpha:.3f} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"PseudoUsed: {use_pseudo} | PseudoN: {num_pseudo} | "
                  f"Src Acc: {acc_src:.4f} | Tgt Acc: {acc_tgt:.4f} | "
                  f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
                  f"Weights: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")


            if f1 > best_f1 and TRAIN_CONFIG['validation']['save_best']:
                best_f1 = f1
                best_metrics = {
                    "epoch": epoch, "acc": acc_tgt,
                    "precision": precision, "recall": recall, "f1": f1
                }


                model_path = os.path.join(PATH_CONFIG['output_dir'], PATH_CONFIG['model_save_name'])
                torch.save(model.state_dict(), model_path)

                pred_path = os.path.join(PATH_CONFIG['output_dir'], PATH_CONFIG['predictions_save_name'])
                np.savetxt(pred_path, pred_tgt.numpy(), fmt='%d')

                cm_path = os.path.join(PATH_CONFIG['output_dir'], PATH_CONFIG['confusion_matrix_save_name'])
                with open(cm_path, "wb") as f:
                    pickle.dump(confusion_matrix(true_tgt, pred_tgt), f)

    # Output final result
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    try:
        cm_path = os.path.join(PATH_CONFIG['output_dir'], PATH_CONFIG['confusion_matrix_save_name'])
        with open(cm_path, "rb") as f:
            conf_matrix = pickle.load(f)

        print("\nBest model performance:")
        print(f"  Epoch: {best_metrics['epoch']}")
        print(f"  Accuracy: {best_metrics['acc']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  F1-score: {best_metrics['f1']:.4f}")

        w1, w2, w3 = model.jump_penalty.get_coefficients()
        print(f"  Jump penalty weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")

    except FileNotFoundError:
        print("Best model was not saved during training")

    print(f"\nModel saved to: {PATH_CONFIG['output_dir']}")


if __name__ == '__main__':
    train_model()
