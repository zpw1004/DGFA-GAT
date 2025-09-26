"""
Configuration File - Retrieves Configuration from Command-Line Arguments
"""


from args import parse_arguments


args = parse_arguments()


DATA_CONFIG = {
    'source_path': args.source_path,
    'target_path': args.target_path,
    'required_cols': ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS", "Facies", "Depth"],
    'feature_cols': ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"],
    'label_col': "Facies",
    'depth_col': "Depth",
    'depth_threshold': args.depth_threshold,
}


GRAPH_CONFIG = {
    'sample_graph': {
        'cluster_method': 'affinity_propagation',
        'damping': 0.7,
        'cluster_distance_percentile': 25,
    },
    'cluster_graph': {
        'n_clusters': args.n_clusters,
        'n_edges': args.n_edges,
    }
}


MODEL_CONFIG = {
    'input_dim': args.input_dim,
    'output_dim': args.output_dim,
    'hidden_dim': args.hidden_dim,
    'heads': args.heads,
}


TRAIN_CONFIG = {
    'device': 'cuda' if args.cuda else 'cpu',
    'epochs': args.epochs,
    'learning_rate': args.lr,
    'optimizer': 'adam',

    'loss': {
        'focal_loss': {
            'gamma': args.focal_gamma,
            'alpha': args.focal_alpha,
        },
        'jump_penalty_weight': args.jump_weight,
        'pseudo_weight': args.pseudo_weight,
    },

    # 伪标签配置
    'pseudo_label': {
        'start_epoch': args.pseudo_start_epoch,
        'interval': args.pseudo_interval,
        'threshold_early': args.pseudo_threshold_early,
        'threshold_late': args.pseudo_threshold_late,
        'late_ratio': args.pseudo_late_ratio,
        'max_per_epoch': args.pseudo_max_per_epoch,
        'max_per_class': args.pseudo_max_per_class,
        'base_weight': 0.3,
    },

    'oversampling': {
        'walk_length': args.walk_length,
        'synth_per_class_to_max': args.synth_per_class,
        'per_node_synth': args.per_node_synth,
    },

    'validation': {
        'save_best': args.save_model,
    }
}


PATH_CONFIG = {
    'output_dir': args.output_dir,
    'model_save_name': 'DGFA-GAT_best_model.pth',
    'predictions_save_name': 'DGFA-GAT_predictions.txt',
    'confusion_matrix_save_name': 'DGFA-GAT_confusion_matrix.pkl',
}


SEED_CONFIG = {
    'seed': args.seed,
}


LOG_CONFIG = {
    'print_freq': args.log_interval,
    'debug': args.debug,
}