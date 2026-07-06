
<div align="center">

<h1>Benchmarking Deepfake Attacks on Deep Face Recognition Systems</h1>

<p>
  <a href="https://doi.org/10.1109/TDSC.2026.3693215">
    <img src="https://img.shields.io/badge/DOI-10.1109%2FTDSC.2026.3693215-blue" alt="DOI">
  </a>
  <img src="https://img.shields.io/badge/IEEE%20TDSC-Accepted-green" alt="IEEE TDSC">
  <img src="https://img.shields.io/badge/Python-3.x-blue" alt="Python">
</p>

<div>
  <a href="https://orcid.org/0009-0004-9742-9360">Shu Peng</a><sup>1</sup>,
  <a href="https://orcid.org/0000-0002-8248-3362">Naipeng Dong</a><sup>1</sup>,
  <a href="https://orcid.org/0000-0002-4267-0745">Wanying Dai</a><sup>2</sup>,
  and <a href="https://orcid.org/0000-0002-6390-9890">Guangdong Bai</a><sup>3</sup>
</div>

<div>
  <sup>1</sup>School of Electrical Engineering and Computer Science, The University of Queensland, Australia
  <br>
  <sup>2</sup>School of Cyber Science and Engineering, Sichuan University, China
  <br>
  <sup>3</sup>Department of Computer Science, City University of Hong Kong, Hong Kong, China
</div>

<h4>
  <a href="https://doi.org/10.1109/TDSC.2026.3693215" target="_blank">[Paper]</a>
  |
  <a href="https://github.com/rps111/Benchmarking-Deepfake-Attacks-on-FR" target="_blank">[Code]</a>
</h4>

</div>

## Updates

* [2026] Paper accepted by **IEEE Transactions on Dependable and Secure Computing (TDSC)**.
* [2026] DOI released: [10.1109/TDSC.2026.3693215](https://doi.org/10.1109/TDSC.2026.3693215).
* [2026] Official benchmark implementation released.

---

## Introduction

This repository is the official implementation of our paper:

**Benchmarking Deepfake Attacks on Deep Face Recognition Systems**
*IEEE Transactions on Dependable and Secure Computing*, 2026.
DOI: [10.1109/TDSC.2026.3693215](https://doi.org/10.1109/TDSC.2026.3693215)

Face recognition systems are widely deployed in identity verification and access control scenarios. However, recent Deepfake generation methods can manipulate facial identity, appearance, and attributes, raising new security concerns for face recognition systems.

This benchmark provides a unified evaluation framework for measuring the impact of diverse Deepfake generation methods on state-of-the-art face recognition systems. The framework evaluates three attack objectives defined in our taxonomy:

* **Direct Impersonation (DI)**: whether a Deepfake sample can impersonate a target identity.
* **Evasion Attack (EA)**: whether a Deepfake sample can bypass recognition and become unidentifiable.
* **Ambiguous Identity Overlap (AIO)**: whether a Deepfake identity can overlap with multiple identities simultaneously.


---

## Framework Overview

The benchmark evaluates Deepfake attacks under a unified gallery-probe setting. Real face images are used to construct the identity gallery, while generated Deepfake images are evaluated as probe samples under different attack objectives.

<!-- 
Optional:
If you have a framework figure, place it under ./figs/overview.png and uncomment the following lines.

<div align="center">
<img src="./figs/overview.png" width="90%">
</div>
-->

---

## Attack Objectives

The evaluation core consists of three specialized Python scripts, each corresponding to one attack objective defined in the paper.

| Script   | Attack Objective               | Description                                                                                                              |
| :------- | :----------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| `DI.py`  | **Direct Impersonation**       | Evaluates whether a Deepfake sample can successfully impersonate a specific target identity to gain unauthorized access. |
| `EA.py`  | **Evasion Attack**             | Measures whether a Deepfake sample can bypass the recognition system and make the subject unidentifiable.                |
| `AIO.py` | **Ambiguous Identity Overlap** | Evaluates whether a Deepfake identity can be simultaneously matched to more than one individual.                         |

---

## Repository Structure

```text
Benchmarking-Deepfake-Attacks-on-FR/
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
├── sensitivity_analysis/
│
├── DI.py
├── EA.py
├── AIO.py
├── environment.yml
└── README.md
```

---

## Dataset Preparation

The benchmark expects two main folders: `gallery/` and `dataset/`.

### Gallery

The `gallery/` folder contains real face images used as reference identities.

```text
gallery/
├── id0/
│   ├── real_01.jpg
│   └── real_02.jpg
├── id1/
│   └── real_01.jpg
└── ...
```

Each subfolder corresponds to one identity.

### Deepfake Dataset

The `dataset/` folder contains generated Deepfake images.

```text
dataset/
├── id0_id1/
│   ├── fake_01.jpg
│   └── fake_02.jpg
├── id1_id2/
│   ├── fake_01.jpg
│   └── fake_02.jpg
└── ...
```

Each subfolder represents an identity pair involved in the Deepfake generation process.

---

## Data Availability

Due to dataset licensing, privacy, and ethical restrictions, this repository does not redistribute real face images or generated Deepfake samples.

Users should prepare the dataset following the structure described above and ensure that their use complies with the corresponding dataset licenses, consent requirements, institutional policies, and ethical guidelines.

---

## Installation

### Download

```bash
git clone https://github.com/rps111/Benchmarking-Deepfake-Attacks-on-FR.git
cd Benchmarking-Deepfake-Attacks-on-FR
```

### Environment

We recommend using Anaconda to manage the Python environment:

```bash
conda env create -f environment.yml
conda activate df-fr-bench
```

---

## Usage

### Direct Impersonation

Direct Impersonation evaluates whether a Deepfake sample can be authenticated as a specific target identity.

```bash
python DI.py \
  --image_dataset_path ./dataset \
  --database_path ./gallery/ \
  --model_name ArcFace \
  --recognition_mode I
```

---

### Evasion Attack

Evasion Attack evaluates whether a Deepfake sample can avoid being correctly recognized as the original subject.

```bash
python EA.py \
  --image_dataset_path ./dataset \
  --database_path ./gallery/ \
  --model_name ArcFace \
  --recognition_mode I
```

---

### Ambiguous Identity Overlap

Ambiguous Identity Overlap evaluates whether a Deepfake identity can simultaneously match multiple identities.

```bash
python AIO.py \
  --image_dataset_path ./dataset \
  --database_path ./gallery/ \
  --model_name Facenet \
  --recognition_mode I
```

---

## Arguments

| Argument               | Description                                               |
| :--------------------- | :-------------------------------------------------------- |
| `--image_dataset_path` | Path to the Deepfake image dataset.                       |
| `--database_path`      | Path to the gallery/reference identity database.          |
| `--model_name`         | Face recognition model used for evaluation.               |
| `--recognition_mode`   | Recognition setting used by the evaluation script.        |

---





## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@article{peng2026benchmarking,
  title={Benchmarking Deepfake Attacks on Deep Face Recognition Systems},
  author={Peng, Shu and Dong, Naipeng and Dai, Wanying and Bai, Guangdong},
  journal={IEEE Transactions on Dependable and Secure Computing},
  pages={1--16},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TDSC.2026.3693215}
}
```

---

## Ethical Use

This repository is released for academic research and reproducibility purposes only.

The benchmark is intended to support the study of Deepfake-related risks to face recognition systems and to encourage the development of more secure and robust identity verification systems.

Users are responsible for ensuring that their use of this code, datasets, and generated samples complies with applicable laws, institutional policies, and ethical requirements.
