# 🩺 Explainable Deep Learning for Glaucoma Detection : 
# A Comparative Analysis of ResNet, Xception, and Vision Transformers


## 📌 Project Overview

This project focuses on automated **glaucoma detection** using retinal fundus images from the **SMDG-19 Multichannel Glaucoma Dataset**.

The objective is to develop an **explainable deep learning pipeline** for binary classification:

- **0 → Non-Glaucoma**
- **1 → Glaucoma**

The current phase includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Transfer learning using CNN architectures
- GPU-accelerated training
- Model evaluation and comparison
- Explainability via Grad-CAM

---

## 🧠 Medical Background

Glaucoma is a progressive optic neuropathy characterized by structural changes in the optic disc and cup region of the retina.

Key clinical indicators include:

- Increased cup-to-disc ratio (CDR)
- Optic nerve head damage
- Neuroretinal rim thinning
- Peripapillary vessel changes

To understand glaucoma detection from a medical perspective, refer to this resource:

🎥 **Glaucoma Fundus Image Explanation (Medical Overview)**  
https://www.youtube.com/watch?v=IZsLJkYFzwg

---

## 📊 Dataset Description

The project uses the **SMDG-19 (South Medical University Multichannel Glaucoma) dataset**, which contains:

- 12,316 fundus images
- 48 metadata features
- Multi-source clinical annotations
- Segmentation masks (partially available)

Although the dataset provides multiple modalities (OCT images, vessel maps, optic disc/cup segmentation masks, clinical variables), this phase focuses exclusively on:

> **Full-color fundus image–based classification**

This decision was made due to high sparsity in auxiliary clinical attributes.

---

## 📈 Exploratory Data Analysis (EDA)

The EDA phase includes:

- Dataset schema inspection
- Label distribution analysis
- Missing value analysis
- Class imbalance evaluation
- Image resolution inspection
- Fundus image visualization
- Path validation and preprocessing

Notebook: EDA_and_Data_Analysis.ipynb

---

## 🧠 Model Architectures

Transfer learning was implemented using pretrained CNN backbones.

### 1️⃣ ResNet50
- ImageNet pretrained weights
- Fine-tuned on fundus dataset
- AdamW optimizer
- Binary classification head

Notebook: train_resnet.ipynb


---

### 2️⃣ Xception
- Depthwise separable convolutions
- ImageNet pretrained weights
- Fine-tuned classifier head
- GPU-accelerated training

Notebook: train_xception.ipynb
---

## 🔍 Explainability

To improve interpretability for medical relevance:

- Grad-CAM heatmaps are generated
- Highlighted regions correspond to optic disc and cup areas
- Model decision transparency is emphasized

This aligns with clinical AI requirements where explainability is critical.

---

## ⚙️ Training Configuration

- Framework: TensorFlow 2.x (Keras API)
- GPU acceleration (CUDA-enabled)
- Custom ImageGenerator class
- Data augmentation (train set only)
- Image normalization (pixel scaling to [0,1])
- Binary Cross-Entropy Loss
- AdamW optimizer
- Stratified dataset splitting

---

## 📊 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (where applicable)

Comparative performance analysis between architectures is part of the methodology.

---

## 🗂 Project Structure
SMDG-Classifier\
│\
├── EDA_and_Data_Analysis.ipynb\
├── resnet.ipynb\
├── xception.ipynb\
├── models.py\
├── requirements.txt\
├── README.md\
└── .gitignore


(Note: Model weights and checkpoints are excluded from version control.)

---

## 🚀 Future Work

Planned extensions include:

- Vision Transformer (ViT) implementation
- CNN vs Transformer comparative analysis
- Multi-task learning (classification + segmentation)
- Integration of clinical metadata features
- Enhanced Grad-CAM visual explanations
- Experiment tracking using Weights & Biases (WandB)
- Streamlit-based interactive diagnostic interface
- Performance benchmarking across architectures

---

## 🧪 Reproducibility

To run this project:

```bash
conda create -n smdg_gpu python=3.10
conda activate smdg_gpu
pip install -r requirements.txt