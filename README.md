# DGFA-GAT
> A Geology-Aware Dual-Graph Attention Network for Cross-Well Lithofacies Interpretation with Limited Labels
- **Comparative Visualization of Lithology Predictions Across Multiple Wells**
<div align="center">

  <!-- 第一行：3 张 -->
  <img src="pic/SHRIMPLIN.jpg" width="300" />
  <img src="pic/LUKE G U.jpg" width="300" />
  <img src="pic/CROSS H CATTLE.jpg" width="300" />

  <!-- 第二行：3 张 -->
  <img src="pic/wb_wa_NOLAN.jpg" width="300" />
  <img src="pic/wb_wa_Recruit F9.jpg" width="300" />
  <img src="pic/wb_wa_NEWBY.jpg" width="300" />

  <!-- 第三行：3 张 -->
  <img src="pic/wb_wa_CHURCHMAN BIBLE.jpg" width="300" />
  <img src="pic/wc_wd_Well1.jpg" width="300" />
  <img src="pic/wd_wc_Well2.jpg" width="300" />

  <!-- 第四行：1 张（居中） -->
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

---

## 🚀 Quick test (required by the journal)

The repository includes a **quick test** that runs end‑to‑end on a tiny synthetic dataset and finishes in under a minute on CPU.

```bash
python train_model.py --src ./dataset/WA.csv --tgt ./dataset/WB1.csv --epochs 80 --out ./runs/exp1
```

This quick test is minimal and intended to verify environment/setup and the CLI interface only. It is **not** meant to reproduce the paper‑level metrics.

---

## 📂 Data

For the quick test we **generate synthetic CSVs** with the following columns (matching the paper):
`GR, ILD_log10, DeltaPHI, PHIND, PE, NM_M, RELPOS, Facies, Depth`.

To run on your real data, prepare two CSV files with the same columns and call:

```bash
python -m dgfa_gat.train --src /path/to/source.csv --tgt /path/to/target.csv --epochs 80 --out runs/exp1
```

> **Reproducibility:** results depend on random seeds and GPU nondeterminism. We set seeds and enable deterministic flags where possible. See code comments for details.

---

## 🧪 Unit/Smoke tests

We include a minimal smoke test in `tests/test_quick.py` that imports the package and exercises one training step on the toy data.

Run:
```bash
pytest -q
```

---

## 📜 License

MIT (see `LICENSE`).

---

## 🔖 How to cite

- This repository ships a `CITATION.cff` file so GitHub renders a *Cite this repository* button.
- We recommend archiving a tagged release on **Zenodo** to obtain a DOI and adding a DOI badge to the README.

---

## ✍️ Computer Code Availability (template for the manuscript)

**Computer Code Availability** — The code developed/used in this study is publicly available at:  
**GitHub:** https://github.com/your-org/dgfa-gat (public, anonymous download)  
**Version archived with DOI:** <add Zenodo DOI here>  
**License:** MIT.  
**Quick‑test:** repository includes `quick_test.py` and instructions in the README.

（中文）**计算机代码可用性声明**：本文使用/开发的代码已在 **GitHub** 公共仓库公开（可匿名下载），并通过 **Zenodo** 存档分配 DOI。仓库包含 README 使用说明、`quick_test.py` 快速测试与 MIT 许可。链接：<填入GitHub与DOI>。

---

## 🧭 Repository layout

```
DGFA-GAT/
├─ CITATION.cff
├─ LICENSE
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ quick_test.py
├─ src/
│  └─ dgfa_gat/
│     ├─ __init__.py
│     ├─ train.py
│     ├─ model.py
│     └─ data_utils.py
├─ data/
│  └─ sample/  # tiny synthetic CSVs for the quick test
└─ tests/
   └─ test_quick.py
```

---

## 📝 Notes for editors/reviewers

- Public repo ✓
- README with instructions ✓
- At least one quick test & how to run ✓
- Open‑source license ✓
- No single compressed file; source is browsable and version‑controlled ✓
