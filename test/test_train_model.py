"""
Test File - Used for verifying the functionality of the code
"""

import torch
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test module imports"""
    print("1. Testing module imports...")
    try:
        from graph_based_da_gat import GraphBasedDA_GAT, FocalLoss, AdaptiveJumpPenalty
        from build_sample_graph import build_sample_graph
        from build_cluster_graph import build_cluster_graph
        from utils import gat_random_walk_oversample, set_seed

        print("   ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("2. Testing model creation...")
    try:
        from graph_based_da_gat import GraphBasedDA_GAT

        model = GraphBasedDA_GAT(input_dim=7)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"   ✓ Model created successfully, number of parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 50)
    print("DGFA-GAT Module Tests")
    print("=" * 50)

    tests = [test_imports, test_model_creation]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test results: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")


if __name__ == "__main__":
    main()
