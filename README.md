# DGFA-GAT
> A Geology-Aware Dual-Graph Attention Network for Cross-Well Lithofacies Interpretation with Limited Labels
- **Comparative Visualization of Lithology Predictions Across Multiple Wells**
<div align="center">


  <img src="pic/SHRIMPLIN.jpg" width="300" />
  <img src="pic/LUKE G U.jpg" width="300" />
  <img src="pic/CROSS H CATTLE.jpg" width="300" />

  <img src="pic/wb_wa_NOLAN.jpg" width="300" />
  <img src="pic/wb_wa_Recruit F9.jpg" width="300" />
  <img src="pic/wb_wa_NEWBY.jpg" width="300" />

  <img src="pic/wb_wa_CHURCHMAN BIBLE.jpg" width="300" />
  <img src="pic/wc_wd_Well1.jpg" width="300" />
  <img src="pic/wd_wc_Well2.jpg" width="300" />

  <p>
    <img src="pic/wd_wc_Well3.jpg" width="300" />
  </p>

</div>

---
Tested with:
- matplotlib==3.7.5
- numpy==1.24.4
- pandas==2.0.3
- scikit-learn==1.3.2
- torch==2.0.0
- torchvision==0.15.0
- torchaudio==2.0.0
- torch-geometric==2.6.1
## 🔧 Installation

```bash
# (1) create and activate a virtual env (example with conda; any env manager is fine)
conda create -n dgfa-gat python=3.9 -y
conda activate dgfa-gat

# (2) install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch/PyG versions**: If you use CUDA, install the matching wheels for your CUDA version (see https://pytorch.org). This repo pins versions that work on CPU and common CUDA setups; adjust if needed.


## 🚀 Quick test (required by the journal)

The repository includes a quick test that verifies the environment setup and basic functionality using synthetic data:

```bash
python test/test_train_model.py
```
---
## 📂 Data

For the quick test we **generate synthetic CSVs** with the following columns (matching the paper):
`GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS, Facies, Depth`.

To run on your real data, prepare two CSV files with the same columns and call:
---
### Basic Training
```bash
python train_model.py \
    --source-path ./dataset/WA.csv \
    --target-path ./dataset/WB1.csv \
    --epochs 80 \
    --output-dir ./runs/exp1
```

> **Reproducibility:** results depend on random seeds and GPU nondeterminism. We set seeds and enable deterministic flags where possible. See code comments for details.

---

## 📜 License

MIT (see `LICENSE`).

---

## ✍️ Computer Code Availability (template for the manuscript)

**Computer Code Availability** — The code developed/used in this study is publicly available at:  
**GitHub:** https://github.com/zpw1004/DGFA-GAT 

**License:** MIT.  
**Quick‑test:** repository includes `quick_test.py` and instructions in the README.

---

## 🧭 Repository layout

```
DGFA-GAT/
├── .gitignore
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── dataset/                  # Geological facies datasets
│   ├── WA.csv               # Source domain data (Well A)
│   └── WB.csv               # Target domain data (Well B)
├── test/                    # Test scripts
│   └── test_train_model.py  # Quick functionality test
├── pic/                     # Experimental results and figures
├── args.py                  # Command-line argument parser
├── config.py                # Configuration management
├── train_model.py           # Main training script
├── graph_based_da_gat.py    # DGFA-GAT model implementation
├── build_sample_graph.py    # Sample graph construction
├── build_cluster_graph.py   # Cluster graph construction
├── utils.py                 # Utility functions
└── __pycache__/             # Python bytecode cache (ignored)
```

---
