# Deep Learning for Medical Imaging (Pneumonia Detection)

## Project Overview
This research initiative investigates the efficacy of **Transfer Learning (ResNet-18)** versus custom CNN architectures for the binary classification of chest X-Ray images (Pneumonia vs. Normal). The primary focus was addressing high **class-imbalance** conditions common in medical datasets.

## Key Features & Methodology
As detailed in the accompanying code, this project implements:

*   **Dynamic Data Augmentation:** Utilized geometric transformations (Rotation, Affine shifts, Horizontal Flips) to improve model generalization.
*   **Weighted Loss Strategy:** Implemented a weighted Binary Cross-Entropy loss function to heavily penalize false negatives, ensuring the model prioritizes sensitivity.
*   **Transfer Learning:** Fine-tuned a pre-trained ResNet-18 architecture on the target dataset.

## Results
*   **Metric Focus:** Recall (Sensitivity) was prioritized over raw accuracy to minimize critical miss rates in diagnosis.
*   **Performance:** The model achieved measurable improvements in Recall compared to baseline custom architectures.

## Tech Stack
*   **Language:** Python
*   **Framework:** PyTorch
*   **Libraries:** Torchvision, OpenCV/PIL, Scikit-Learn

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install torch torchvision torchmetrics`
3. Download the Chest X-Ray dataset (e.g., from Kaggle) and place it in `data/chestxrays`.
4. Run the notebook `Pneumonia_Detection.ipynb`.


