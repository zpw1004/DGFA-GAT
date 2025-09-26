"""
命令行参数解析
"""

import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='DGFA-GAT: Dual-Graph Fusion Adaptive Graph Attention Network')

    # =========================
    # 基本配置
    # =========================
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=270,
                        help='Random seed.')

    # =========================
    # 数据路径配置
    # =========================
    parser.add_argument('--source-path', type=str, default="dataset/WA.csv",
                        help='Path to source domain data')
    parser.add_argument('--target-path', type=str, default="dataset/WB.csv",
                        help='Path to target domain data')
    parser.add_argument('--output-dir', type=str, default="pseudo_out",
                        help='Directory to save output files')

    # =========================
    # 训练配置
    # =========================
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')

    # =========================
    # 模型配置
    # =========================
    parser.add_argument('--input-dim', type=int, default=7,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--heads', type=int, default=32,
                        help='Number of attention heads')

    # =========================
    # 图构建配置
    # =========================
    parser.add_argument('--depth-threshold', type=float, default=0.5,
                        help='Depth threshold for building edges')
    parser.add_argument('--n-clusters', type=int, default=9,
                        help='Number of clusters for cluster graph')
    parser.add_argument('--n-edges', type=int, default=20,
                        help='Number of edges per cluster')

    # =========================
    # 损失函数配置
    # =========================
    parser.add_argument('--focal-gamma', type=float, default=3.0,
                        help='Gamma parameter for Focal Loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Alpha parameter for Focal Loss')
    parser.add_argument('--jump-weight', type=float, default=0.2,
                        help='Weight for jump penalty loss')
    parser.add_argument('--pseudo-weight', type=float, default=0.4,
                        help='Weight for pseudo label loss')

    # =========================
    # 伪标签配置
    # =========================
    parser.add_argument('--pseudo-start-epoch', type=int, default=10,
                        help='Epoch to start using pseudo labels')
    parser.add_argument('--pseudo-interval', type=int, default=5,
                        help='Interval for using pseudo labels')
    parser.add_argument('--pseudo-threshold-early', type=float, default=0.9,
                        help='Confidence threshold for early stage')
    parser.add_argument('--pseudo-threshold-late', type=float, default=0.8,
                        help='Confidence threshold for late stage')
    parser.add_argument('--pseudo-late-ratio', type=float, default=0.6,
                        help='Ratio to switch to late stage threshold')
    parser.add_argument('--pseudo-max-per-epoch', type=int, default=64,
                        help='Maximum pseudo labels per epoch')
    parser.add_argument('--pseudo-max-per-class', type=int, default=10,
                        help='Maximum pseudo labels per class')

    # =========================
    # 过采样配置
    # =========================
    parser.add_argument('--walk-length', type=int, default=1,
                        help='Random walk length for oversampling')
    parser.add_argument('--no-synth-per-class', action='store_false', dest='synth_per_class',
                        help='Disable synthesize to max class count')
    parser.set_defaults(synth_per_class=True)
    parser.add_argument('--per-node-synth', type=int, default=1,
                        help='Synthetic samples per node')

    # =========================
    # 实验配置
    # =========================
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode')
    parser.add_argument('--no-save-model', action='store_false', dest='save_model',
                        help='Disable model saving')
    parser.set_defaults(save_model=True)
    parser.add_argument('--log-interval', type=int, default=1,
                        help='Log interval in epochs')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.output_dim = 9  # 固定输出类别数

    return args