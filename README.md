# DGFA-GAT (Reproducible Research Release)

This repository contains the reference implementation for our manuscript (working title):

> _Domain-Generalized Fusion-Attention GAT for Well-Log Facies Classification_

It is prepared to comply with **Computers & Geosciences** pre‑review requirements for the *Computer Code Availability* section: public repository, README with run instructions, at least one quick test, and an open-source license.

---

## 🔧 Installation

```bash
# (1) create and activate a virtual env (example with conda; any env manager is fine)
conda create -n dgfa-gat python=3.10 -y
conda activate dgfa-gat

# (2) install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch/PyG versions**: If you use CUDA, install the matching wheels for your CUDA version (see https://pytorch.org). This repo pins versions that work on CPU and common CUDA setups; adjust if needed.

---

## 🚀 Quick test (required by the journal)

The repository includes a **quick test** that runs end‑to‑end on a tiny synthetic dataset and finishes in under a minute on CPU.

```bash
# option A: one-liner quick test
python quick_test.py

# option B: generate toy CSVs + run the training entry point
python quick_test.py --only-generate
python -m dgfa_gat.train --src data/sample/source_domain.csv --tgt data/sample/target_domain.csv --epochs 3 --out out_quick
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
