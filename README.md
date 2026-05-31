# Multimodal Disaster Intelligence System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20LightGBM-orange?logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![Conference](https://img.shields.io/badge/Published-ICRETM%202026-blueviolet)
![Stars](https://img.shields.io/github/stars/SHANMUGAAPRIYANM/Multimodal-Disaster-Intelligence-System?style=social)

An autonomous multimodal AI framework for disaster prediction, satellite flood segmentation, physics-informed validation, ensemble fusion, and adaptive monitoring across earthquake, cyclone, wildfire, and flood events.

> 📄 **Research paper published in proceedings of ICRETM 2026** — 6th International Conference on Recent Trends in Engineering Technology and Management.

---

## Table of Contents

- [Overview](#1-overview)
- [Problem Statement](#2-problem-statement)
- [Architecture](#3-architecture)
- [Repository Structure](#repository-structure)
- [Datasets](#4-datasets)
- [Models Used](#5-models-used)
- [Training Pipeline](#6-training-pipeline)
- [Evaluation Metrics](#7-evaluation-metrics)
- [Results](#8-results)
- [Screenshots / Outputs](#9-screenshots--outputs)
- [Future Improvements](#10-future-improvements)
- [Tech Stack](#11-tech-stack)
- [How to Run](#12-how-to-run)
- [Conclusion](#conclusion)

---

## 1. Overview

Natural disasters such as earthquakes, floods, cyclones, and wildfires cause severe human, economic, and environmental losses every year. Existing disaster prediction systems are often limited to single-modality analysis and lack adaptive intelligence mechanisms for continuous monitoring and self-improvement.

The proposed **Multimodal Disaster Intelligence System** presents an autonomous AI-driven framework that integrates environmental data, climate information, seismic records, and satellite imagery into a unified multimodal prediction pipeline. The system combines multiple specialist machine learning models, deep learning-based satellite segmentation, ensemble fusion learning, physics-informed validation, and adaptive monitoring mechanisms to improve disaster prediction reliability.

The framework incorporates:

- Earthquake prediction using seismic features
- Wildfire prediction using climate and fire weather data
- Cyclone prediction using atmospheric parameters
- Flood prediction using hydrological and meteorological data
- Satellite flood segmentation using Sentinel-2 imagery
- Autonomous monitoring using drift detection and AI-based reasoning

The proposed system aims to move beyond static prediction pipelines and towards adaptive AI systems capable of monitoring environmental changes and responding intelligently to data drift and evolving disaster patterns.

---

## 2. Problem Statement

Traditional disaster prediction systems suffer from several major limitations:

- Single-modality learning approaches
- Lack of real-time adaptation
- Poor integration of satellite imagery and tabular data
- Limited scalability across disaster domains
- Absence of autonomous monitoring mechanisms
- High false positive and false negative rates in safety-critical environments

Most existing systems are designed specifically for a single disaster type and fail to leverage multimodal environmental intelligence. In addition, environmental data distributions continuously evolve over time due to climate variability, sensor changes, and geographic shifts. Static AI systems degrade in performance when exposed to unseen conditions.

The proposed project addresses these challenges by developing a unified multimodal disaster intelligence framework that combines:

- Multiple specialist AI models
- Satellite image segmentation
- Ensemble fusion learning
- Physics-informed validation
- Drift-aware autonomous monitoring

The objective is to improve prediction accuracy, robustness, scalability, and long-term reliability for real-world disaster management applications.

---

## 3. Architecture

The proposed architecture consists of multiple interconnected modules operating in a closed-loop intelligent prediction pipeline.

The overall workflow includes:

1. Data acquisition from multiple disaster datasets
2. Data preprocessing and feature engineering
3. Specialist disaster prediction models
4. Satellite flood segmentation
5. Fusion meta-learning
6. Physics-informed validation
7. Autonomous monitoring and drift detection

---

### High-Level System Architecture

![System Architecture](Architecture%20Diagram%20/Figure%203%201.PNG)

---

### Multimodal Fusion Architecture

![Fusion Architecture](Architecture%20Diagram%20/Figure%203%203.PNG)

---

### Autonomous AI Monitoring Workflow

![Monitoring Workflow](Architecture%20Diagram%20/autonomous_monitoring.png)

---

### Physics-Informed Validation Layer

![Physics Gate](Architecture%20Diagram%20/Figure%203%208.PNG)

---

## Repository Structure

```
📁 Multimodal-Disaster-Intelligence-System/
│
├── 📁 Source/
│   └── Multimodal_Disaster_Intelligence_System.ipynb   # Full end-to-end pipeline notebook
│       ├── Section 1  — Data Ingestion & Preprocessing
│       ├── Section 2  — Feature Engineering
│       ├── Section 3  — Earthquake Specialist LightGBM
│       ├── Section 4  — Wildfire Specialist LightGBM
│       ├── Section 5  — Cyclone Specialist LightGBM
│       ├── Section 6  — Flood Specialist LightGBM
│       ├── Section 7  — Sentinel-2 Satellite Flood Segmentation (UNet & SegFormer-Lite)
│       ├── Section 8  — Physics-Informed Hybrid Scoring (Saffir-Simpson & FWI Gates)
│       ├── Section 9  — Leakage-Free Fusion Meta-LightGBM
│       ├── Section 10 — Multimodal Prediction Output
│       └── Section 11 — Autonomous AI Scientist Monitoring Loop (KL-Divergence)
│
├── 📁 Architecture Diagram/
│   ├── Figure 3 1.PNG     — High-level system architecture
│   ├── Figure 3 3.PNG     — Multimodal fusion architecture
│   ├── Figure 3 8.PNG     — Physics-informed validation layer
│   └── autonomous_monitoring.png — Autonomous monitoring workflow
│
├── requirements.txt        # All Python dependencies with pinned versions
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 4. Datasets

The system utilizes multiple real-world open-source datasets collected from internationally recognized scientific organizations.

| Dataset | Purpose | Source | Records |
|---|---|---|---|
| USGS Earthquake Dataset | Earthquake prediction | USGS | 112,291 |
| IBTrACS v04r00 | Cyclone prediction | NOAA | 291,780 |
| NASA FIRMS VIIRS | Wildfire prediction | NASA | 2,153 events |
| ERA5 / Open-Meteo | Climate and weather analysis | ECMWF | 10,958 daily |
| WorldFloods v1.0 | Flood segmentation | SpaceML | 37 images |
| EM-DAT | Disaster ground truth labeling | EM-DAT | 16,747 records |

---

### Dataset Characteristics

#### Earthquake Dataset
- 112,291 seismic records (M ≥ 4.5, global, 2010–2024)
- Features: magnitude, depth, latitude, longitude, energy, ring-of-fire flag

#### Cyclone Dataset
- 291,780 cyclone observations from IBTrACS
- Features: wind speed, pressure drop, SST proxy, cyclone energy

#### Wildfire Dataset
- Active fire observations from NASA FIRMS VIIRS
- Features: fire radiative power, brightness temperature, detection confidence

#### Flood Dataset
- Hydrological and precipitation features (Open-Meteo ERA5)
- Multi-temporal rainfall accumulation: 3-day, 7-day, 30-day windows

#### Satellite Dataset
- Sentinel-2 NIR + SWIR1 multispectral imagery
- Flood masks from WorldFloods v1.0 (37 real images)

---

## 5. Models Used

The system combines multiple machine learning and deep learning models.

| Model | Purpose |
|---|---|
| LightGBM (×4 specialists) | Specialist disaster prediction |
| UNet (7.76M params) | Flood image segmentation |
| SegFormer-Lite (2.80M params) | Lightweight flood segmentation |
| Fusion Meta-LightGBM | Ensemble fusion learning |
| Optuna | Hyperparameter optimization (50 trials, 5-fold CV) |
| KL-Divergence Monitor | Distribution shift detection |

---

### LightGBM Specialist Models

Four independent LightGBM specialist models are trained for:

- Earthquake prediction
- Wildfire prediction
- Cyclone prediction
- Flood prediction

LightGBM is selected because of:
- High accuracy on tabular data
- Fast training speed with histogram-based split finding
- Strong generalization on limited samples
- Effective handling of class imbalance with SMOTE

---

### UNet Segmentation Model

UNet (7.76M parameters) is used for flood-region segmentation from Sentinel-2 satellite images, using a symmetric encoder-decoder architecture with skip connections.

Loss function: `L = 0.5 × BCE + 0.5 × Dice`

UNet was selected because:
- Works efficiently on small datasets (25 training images)
- Strong spatial localization via skip connections
- Proven performance in remote sensing segmentation tasks

---

### Physics-Informed Hybrid Scoring

A physics gate is applied after ML scoring using:
- **Saffir-Simpson scale** for cyclone confidence floors (Category 1–5)
- **Fire Weather Index (FWI)** for wildfire confidence floors

```
score_final = max(P_ML(y|x), floor_physics(x))
```

This prevents ML uncertainty from suppressing alerts for physically unambiguous extreme events.

---

### Fusion Meta-Learning

Outputs from all specialist models are combined using a leakage-free Fusion Meta-LightGBM.

The 9-dimensional meta-feature vector:
```
φ(x) = [P_EQ, P_WF, P_CY, P_FL, P_CY·P_FL, P_EQ·P_FL, max_d, mean_d, std_d]
```

Interaction features explicitly capture compound disaster signals (cyclone-induced flooding, earthquake-triggered tsunamis).

---

## 6. Training Pipeline

### Step 1: Data Collection
Environmental, climate, seismic, and satellite datasets collected from public repositories (USGS, IBTrACS, NASA FIRMS, Open-Meteo, WorldFloods).

### Step 2: Data Preprocessing
- Missing value handling
- Normalization and temporal alignment
- Feature extraction
- Ground truth label generation from EM-DAT (zero data leakage design)

### Step 3: Feature Engineering

| Specialist | Features (n) | Key Engineered Features |
|---|---|---|
| Earthquake | 11 | mag², log(depth), log(energy), ring-of-fire, recent quake count |
| Wildfire | 5 | Brightness temp, FRP, detection confidence, lat/lon |
| Cyclone | 9 | Pressure drop ΔP, wind², SST proxy, CE = V²·SST |
| Flood | 10 | Precip 3d/7d/30d, snowmelt spike, wind, temp |

### Step 4: Specialist Model Training
- Stratified 80/20 train-test split
- SMOTE oversampling on training split only
- Optuna hyperparameter optimization (50 trials, 5-fold stratified CV)

### Step 5: Satellite Segmentation Training
- UNet and SegFormer-Lite trained on 25 Sentinel-2 images, validated on 5, tested on 7
- Training augmentations: RandomRotate90, HorizontalFlip, VerticalFlip, GaussNoise, ElasticTransform
- Optimizer: AdamW (lr=1e-3, weight decay=1e-4), CosineAnnealingLR over 20 epochs

### Step 6: Fusion Learning
- Specialist predictions on held-out 20% test splits form meta-features
- Realistic background probability distributions for inactive modalities prevent leakage
- Multimodal flood combiner: `P_fused = 0.5 × P_tabular + 0.5 × P_satellite`

### Step 7: Autonomous Monitoring
Continuous monitoring using:
- KL-Divergence: `D_KL(P ∥ Q) = Σ P(xᵢ) · log[P(xᵢ)/Q(xᵢ)]`
- Drift threshold δ = 0.10; F1 drop threshold ε = 0.05
- Selective retraining of only affected specialist(s)

---

## 7. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions |
| Recall | Disaster detection sensitivity |
| F1 Score | Balance between precision and recall |
| AUC-ROC | Class separability across thresholds |
| IoU | Flood segmentation spatial overlap quality |

---

## 8. Results

### Specialist Model Performance

| Model | Accuracy | F1 Score | AUC-ROC | Precision | Recall | Events |
|---|---|---|---|---|---|---|
| Earthquake | 88.92% | 89.99% | 0.9451 | 88.5% | 91.6% | 11,680 |
| Wildfire | 82.13% | 82.00% | 0.8898 | 83.1% | 80.9% | 2,153 |
| Cyclone | 96.01% | 96.08% | **0.9920** | 96.3% | 95.8% | 34,409 |
| Flood | 93.52% | 93.53% | **0.9847** | 93.6% | 93.5% | 3,574 |

### Key Observations

- Cyclone prediction achieved the highest performance with **AUC-ROC = 0.9920** — attributed to the strong physical separability of cyclone events in wind/pressure feature space
- Flood prediction demonstrated strong generalization using multi-temporal precipitation accumulation windows
- Earthquake prediction prioritized high recall (91.6%) to minimize missed disaster events
- Wildfire performance affected by spatial label noise (FIRMS point-level vs EM-DAT country-month)

### Fusion Meta-Learning Results

- **5-Fold Cross-Validated Weighted F1 = 0.8004** (leakage-free, honest evaluation)
- Strong multimodal class separability confirmed via meta-feature separability matrix
- Compound disaster scenarios (cyclone + flood, earthquake + tsunami) handled via interaction features

### Satellite Segmentation Results

| Model | Parameters | Accuracy | IoU | F1 Score |
|---|---|---|---|---|
| **UNet** | 7.76M | **92.57%** | **11.47%** | **20.58%** |
| SegFormer-Lite | 2.80M | 81.84% | 4.64% | 8.88% |

UNet outperforms SegFormer-Lite confirming CNN inductive bias advantages on small satellite datasets (25 training images).

### Autonomous Monitoring Results

| Cycle | Drift D_KL | EQ F1 | WF F1 | CY F1 | FL F1 | Action |
|---|---|---|---|---|---|---|
| 1 | 0.0078 | 0.971 | 0.702 | 0.948 | 1.000 | Stable — No retraining |
| 2 | 0.0490 | 0.971 | 0.702 | 0.948 | 1.000 | Stable — No retraining |
| 3 | 0.0157 | 0.971 | 0.702 | 0.948 | 1.000 | Stable — No retraining |

All drift magnitudes remained well below threshold δ = 0.10, confirming model stability.

---

## 9. Screenshots / Outputs

### Prediction Output
![Prediction Output](Architecture%20Diagram%20/prediction_output.png)

### Satellite Segmentation Output
![Segmentation 1](Architecture%20Diagram%20/segmentation_1.png)
![Segmentation 2](Architecture%20Diagram%20/segmentation_2.png)
![Segmentation 3](Architecture%20Diagram%20/segmentation_3.png)

### Drift Detection Monitoring
![Drift 1](Architecture%20Diagram%20/drift_1.png)
![Drift 2](Architecture%20Diagram%20/drift_2.png)
![Drift 3](Architecture%20Diagram%20/drift_3.png)
![Drift 4](Architecture%20Diagram%20/drift_4.png)
![Drift 5](Architecture%20Diagram%20/drift_5.png)

---

## 10. Future Improvements

Future enhancements may include:

- Real-time disaster streaming pipelines (Apache Kafka + Flink)
- IoT sensor integration for ground-truth validation
- Transformer-based multimodal learning (CLIP, ViT-based fusion)
- Physics-Informed Neural Networks (PINNs) for geophysical modelling
- Cloud deployment on AWS / GCP with REST API
- Mobile disaster alert systems
- Explainable AI (XAI) — SHAP, LIME for model interpretability
- Reinforcement learning for autonomous adaptation strategies
- Expansion to tsunami, landslide, and volcanic eruption prediction

The framework can eventually evolve into a large-scale autonomous environmental intelligence platform for global disaster monitoring.

---

## 11. Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10+ |
| ML Framework | LightGBM, Scikit-learn |
| Deep Learning | PyTorch, segmentation-models-pytorch |
| Image Processing | OpenCV, Albumentations, Rasterio |
| Visualization | Matplotlib, Seaborn, Plotly |
| Hyperparameter Optimization | Optuna |
| Class Balancing | SMOTE (imbalanced-learn) |
| Platform | Google Colab Pro |
| Monitoring | KL-Divergence, Groq API |
| Version Control | GitHub |

---

## 12. How to Run

### Clone Repository

```bash
git clone https://github.com/SHANMUGAAPRIYANM/Multimodal-Disaster-Intelligence-System.git
cd Multimodal-Disaster-Intelligence-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

Open `Source/Multimodal_Disaster_Intelligence_System.ipynb` using:

- **Google Colab** (recommended — GPU required for segmentation training)
- **Jupyter Notebook** with GPU support

```bash
jupyter notebook Source/Multimodal_Disaster_Intelligence_System.ipynb
```

### Execute Pipeline

Run all notebook cells sequentially to:

1. Ingest and preprocess all 6 datasets
2. Engineer disaster-specific features
3. Train 4 specialist LightGBM models (with SMOTE + Optuna)
4. Train UNet satellite segmentation on WorldFloods v1.0
5. Apply physics-informed hybrid scoring (Saffir-Simpson + FWI)
6. Train leakage-free Fusion Meta-LightGBM
7. Generate multimodal disaster predictions
8. Execute autonomous monitoring loop (KL-divergence drift detection)

> ⚠️ **Note:** Satellite segmentation (UNet/SegFormer-Lite) requires GPU. Use Google Colab Pro or a CUDA-enabled environment. All other pipeline components run on CPU.

---

## Conclusion

The proposed Multimodal Disaster Intelligence System demonstrates the effectiveness of combining multimodal AI, satellite segmentation, ensemble learning, and autonomous monitoring for disaster prediction. By integrating structured environmental data with computer vision and adaptive AI workflows, the framework moves toward intelligent and scalable disaster management systems capable of supporting future climate resilience applications.

Specialist models achieve AUC-ROC values from **0.8898 to 0.9920**, with the leakage-free Fusion Meta-LightGBM achieving **CV weighted F1 = 0.8004**. The Autonomous AI Scientist loop correctly maintained stable models across all evaluation cycles with no unnecessary retraining.

---

## Citation

If you use this work, please cite:

```
Shanmugaapriyan M, Dr. R.S. Ponmagal,
"An Autonomous AI Scientist for Planet-Scale Disaster Prediction Using Multimodal 
Satellite and Geophysical Data Fusion",
Proceedings of ICRETM 2026 — 6th International Conference on Recent Trends in 
Engineering Technology and Management, 2026.
```

---

## Author

**Shanmugaapriyan M**
M.Tech — Artificial Intelligence & Data Science
SRM Institute of Science and Technology, Kattankulathur

[![GitHub](https://img.shields.io/badge/GitHub-SHANMUGAAPRIYANM-black?logo=github)](https://github.com/SHANMUGAAPRIYANM)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/shanmugaapriyan-m-b6529a18a)
[![Email](https://img.shields.io/badge/Email-shanmugaapriyan04%40gmail.com-red?logo=gmail)](mailto:shanmugaapriyan04@gmail.com)

---

*Licensed under the [MIT License](LICENSE)*
