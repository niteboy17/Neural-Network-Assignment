# CNN Assignment - Chest X-Ray Pneumonia Classification

## Problem Statement

The dataset comprises 5,863 JPEG X-Ray images and is categorised into two types:

- Pneumonia
- Normal

The images are organised into three folders:

- `train`
- `test`
- `val`

Each folder contains subfolders for the two image categories:

- `PNEUMONIA`
- `NORMAL`

These chest X-ray images are anterior-posterior views sourced from paediatric patients aged one to five years at a renowned hospital. The X-rays were part of the routine clinical care of these patients.

As a data scientist working in the healthcare industry, the objective is to develop a **Convolutional Neural Network (CNN)** model that can classify chest X-ray images into their respective categories based on the image itself.

> **Note:** This project is for academic purposes and should not be considered a clinically validated diagnostic system.

---

# Dataset

The dataset is structured as follows:

```text
x-ray-dataset/
│
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
│
├── val/
│   ├── PNEUMONIA/
│   └── NORMAL/
│
└── test/
    ├── PNEUMONIA/
    └── NORMAL/