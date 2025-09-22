# **VARDiff: Vision-Augmented Retrieval-Guided Diffusion for Stock Forecasting**

## 📌 Overview
VARDiff is a novel **vision-guided diffusion framework** for **uncertainty-aware stock forecasting**, combining the complementary strengths of diffusion models and vision-based retrieval.

- Historical time series are transformed into **image representations** and embedded using a **pretrained vision encoder** to capture rich spatial features.  
- Using **cosine similarity matching**, we retrieve semantically similar historical patterns that serve as **conditional guidance** during the diffusion denoising process.  
- This **retrieval-guided conditioning mechanism** enables the model to generate **more accurate and contextually-informed forecasts**, while producing **well-calibrated predictive distributions** to better quantify uncertainty.  

<p align="center">
  <img src="visual/overview.png" alt="VARDiff Overview" width="600">
</p>

---

## ⚙️ Setup

Clone the repository:
```bash
git clone https://github.com/AppliedAI-Lab/VARDiff.git
cd VARDiff
```
Install dependencies:
We provide a requirements.yaml file for Conda environment configured to run the model:
```bash
conda env create -f requirements.yaml
conda activate VARDiff
```

🚀 Usage
🔹 Retrieval Process (to create database)
```bash
cd retrieval
python univariate_embedding.py
```
Or using the provided script:
```bash
cd scripts
./retriever.sh
```
🔹 Diffusion Process (to generate forecasts)

Run with default settings or tune hyperparameters:
```bash
cd scripts
./diffusion.sh
```
📖 Citation

If you find this work useful, please consider citing:
@article{vardiff2025,
  title={VARDiff: Vision-Augmented Retrieval-Guided Diffusion for Stock Forecasting},
  author={Your Name and Others},
  year={2025},
  journal={arXiv preprint arXiv:xxxx.xxxxx}
}



