# Multimodal-Disaster-Intelligence-System
An autonomous multimodal AI framework for disaster prediction, satellite flood segmentation, physics-informed validation, ensemble fusion, and adaptive monitoring across earthquake, cyclone, wildfire, and flood events.

# Multimodal Disaster Intelligence System

## Overview
— Natural disasters cause immense loss of life and infrastructure annually. Existing prediction systems are typically task-specific, single-modality, and rely on manually engineered pipelines, limiting their adaptability. This paper presents an Autonomous AI Scientist for planet-scale disaster prediction that integrates four modality-specific LightGBM specialist models, a Sentinel-2 optical satellite segmentation module (UNet and SegFormer-Lite), a Fusion Meta-LightGBM learner, and an Autonomous AI Scientist monitoring loop into a unified closed-loop framework. Specialist models are trained on real-world datasets — USGS earthquakes (112,291 records), IBTrACS cyclones (291,780 records), NASA FIRMS wildfire observations (2,153 events), and Open-Meteo ERA5 climate reanalysis (10,958 records) — with ground truth labels derived from EM-DAT (16,747 disaster records, 2010–2024). Satellite flood segmentation uses 37 real Sentinel-2 images from the WorldFloods v1.0 dataset. The specialist models achieve AUC-ROC values of 0.9451 (earthquake), 0.8898 (wildfire), 0.9920 (cyclone), and 0.9847 (flood). The Fusion Meta-LightGBM achieves a 5-fold cross-validated weighted F1 of 0.8004, trained without data leakage using held-out specialist predictions. A physics-informed hybrid scoring mechanism provides safety floors derived from the Saffir-Simpson scale and Fire Weather Index. The Autonomous AI Scientist loop monitors distributional drift via KL-divergence and triggers selective retraining without human intervention. Results demonstrate that the proposed multimodal architecture outperforms single-modality baselines and represents a significant step toward self-directed AI systems for global-scale disaster risk reduction.

## Features
- Earthquake prediction
- Wildfire prediction
- Flood segmentation
- Autonomous monitoring

## Architecture
<img width="283" height="569" alt="Figure 3 2" src="https://github.com/user-attachments/assets/ff443311-a33a-4a03-b390-8e63d74e9a23" />
<img width="549" height="426" alt="Figure 3 3" src="https://github.com/user-attachments/assets/94f7bbba-9088-4cbf-b45a-cf86ac43bc44" />
<img width="1536" height="1024" alt="Figure 3 8" src="https://github.com/user-attachments/assets/75156e30-5e03-44e9-93de-96651e1725de" />
<img width="565" height="331" alt="image" src="https://github.com/user-attachments/assets/81fc75c6-46a9-411a-affa-c7bce2e3cbff" />



## Datasets Used
- USGS
- NASA FIRMS
- IBTrACS
- WorldFloods

## Algorithms
- LightGBM
- UNet
- Fusion Meta Learning
- Optuna

## Results
<img width="569" height="338" alt="image" src="https://github.com/user-attachments/assets/23159936-fd0a-453d-b4ed-7397bdd50eac" />
## Specialist Model Performance

| Model | Accuracy | F1 Score | AUC-ROC | Precision | Recall | Events |
|---|---|---|---|---|---|---|
| Earthquake | 88.92% | 89.99% | 0.9451 | 88.5% | 91.6% | 11,680 |
| Wildfire | 82.13% | 82.00% | 0.8898 | 83.1% | 80.9% | 2,153 |
| Cyclone | 96.01% | 96.08% | 0.9920 | 96.3% | 95.8% | 34,409 |
| Flood | 93.52% | 93.53% | 0.9847 | 93.6% | 93.5% | 3,574 |

### Key Observations

- The **Cyclone** specialist achieved the highest performance with an **AUC-ROC of 0.9920** and **F1 Score of 96.08%**, indicating excellent class separability and prediction capability.
- The **Flood** model also demonstrated strong performance with an **Accuracy of 93.52%** and **AUC-ROC of 0.9847**.
- The **Earthquake** model maintained high recall (**91.6%**) to minimize missed disaster events in safety-critical scenarios.
- The **Wildfire** specialist showed comparatively lower performance due to spatial mismatch and label noise in wildfire datasets.

## Future Scope
