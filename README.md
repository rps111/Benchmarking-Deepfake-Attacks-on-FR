# Benchmarking-Deepfake-Attacks-on-FR

Official implementation for evaluating the impact of diverse Deepfake generation methods on state-of-the-art face recognition (FR) systems. 

This repository provides a unified benchmarking framework to evaluate three distinct attack objectives defined in our taxonomy: **Direct Impersonation (DI)**, **Evasion Attack (EA)**, and **Ambiguous Identity Overlap (AIO)**


---

## Repository Structure

The evaluation core consists of three specialized Python scripts, each corresponding to an attack objective defined in the paper:

| Script | Attack Objective | Description |
| :--- | :--- | :--- |
| `DI.py` | **Direct Impersonation** | Evaluates if a Deepfake can successfully impersonate a specific target identity to gain unauthorized access. |
| `EA.py` | **Evasion Attack** | Measures the ability to bypass recognition systems, rendering the subject unidentifiable. |
| `AIO.py` | **Ambiguous Identity Overlap** | Evaluates if a Deepfake identity can pass authentication for more than one individual simultaneously. |

## Dataset Structure
```text
Deepfake-FR-Bench/
│
├── gallery/
│   ├── id0/                       
│   │   ├── real_01.jpg
│   │   └── real_02.jpg
│   ├── id1/                       
│   │   └── real_01.jpg
│   └── ...
│
├── dataset/
│   ├── id0_id1/                   
│   │   ├── fake_01.jpg
│   │   └── fake_02.jpg
│   ├── id1_id2/                   
│   │   ├── fake_01.jpg
│   │   └── fake_02.jpg
│   └── ...
│
├── DI.py
├── EA.py
└── AIO.py
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rps111/Benchmarking-Deepfake-Attacks-on-FR.git
   cd Deepfake-FR-Bench
    ```
## Usage

All scripts support the --threshold_param argument to scale default decision thresholds (e.g., 1.0 for default, 0.8 for stricter).

Direct Impersonation (DI)
```bash
python DI.py --image_dataset_path ./dataset --database_path ./gallery/ --model_name ArcFace --threshold_param 1.0
```


Evasion Attack (EA)
```bash
python EA.py --image_dataset_path ./dataset --database_path ./gallery/ --model_name ArcFace --threshold_param 1.0
```

Ambiguous Identity Overlap (AIO)
```bash
python AIO.py --image_dataset_path ./dataset --database_path ./gallery/ --model_name Facenet --threshold_param 1.0
```
