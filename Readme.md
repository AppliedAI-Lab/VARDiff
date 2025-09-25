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

## 🚀 Usage  

A **quick & visually appealing guide** to run the **Retrieval → Diffusion** pipeline for both *univariate* and *multivariate* time series.  

---

### 🔹 Retrieval Process (Build Reference Database)

#### 📈 Univariate Time Series (e.g., stock datasets in this paper)
```bash
cd retrieval
python univariate_embedding.py \
  --symbol_list <desired_dataset> \
  --his_len_list 20 40 60 80 100 \
  --step_size_list 5 \
  --num_first_layers 4
```
**Notes:**  
• `symbol_list` → list of datasets/symbols *(9 symbols in this paper)*  
• `his_len_list` → historical lengths for benchmark *(future length = historical length)*  
• `num_first_layers` → number of first layers from pretrained vision encoder  
• `step_size_list` → step sizes *(details in Section 6.4 of the paper)*  
• ⚡ Default: number of retrieved references k = 10 by default because it can reuser for smaller cases)

Or simply use the provided script:
```bash
cd scripts
./retriever.sh
```

📊 Multivariate Time Series (e.g., ETT dataset)

We implement independent feature retrieval:
```bash
cd retrieval
python multivariate_embedding.py \
  --symbol <desired_dataset> \
  --his_len_list 20 40 60 80 100 \
  --step_size_list 5 \
  --num_first_layers 4

```
🔹 Diffusion Process (to generate forecasts)

▶️ Run on a specific dataset

Works for both univariate & multivariate:
```bash
python run_conditional.py --config ./configs/extrapolation/<desired_dataset>.yaml 
```
⚙️ Moreover, we can un with default settings / tune hyperparameters
```bash
cd scripts
./diffusion.sh
```
## 📖 Citation
```bash
If you find this work useful, please consider citing:
@article{vardiff2025,
  title={VARDiff: Vision-Augmented Retrieval-Guided Diffusion for Stock Forecasting},
  author={N.T.Thu, T.X.Thong, N.K.T.Binh, N.N.Hai},
  organization={HUST}
  year={2025},
  journal={Information Sciences}
}
```


