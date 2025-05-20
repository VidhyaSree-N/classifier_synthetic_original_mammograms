# 🩻 Mammogram Classifiers

This repository features two deep learning-based image classifiers tailored for mammogram analysis:

- **[Classifier 1: Synthetic vs. Original Mammogram Detection](https://github.com/VidhyaSree-N/classifier_synthetic_original_mammograms/blob/main/Classifier_simplified.ipynb)**
- **[Classifier 2: Cancerous vs. Non-Cancerous Mammogram Detection](https://github.com/VidhyaSree-N/classifier_synthetic_original_mammograms/blob/main/C_Classifier.ipynb)**

These models are designed to support research in medical imaging and aid in improving diagnostic workflows by automating classification tasks on mammographic data.

---

## 🧠 Classifier 1: Synthetic vs. Original Mammogram Detection

📍 **Notebook Link**: [Classifier_simplified.ipynb](https://github.com/VidhyaSree-N/classifier_synthetic_original_mammograms/blob/main/Classifier_simplified.ipynb)

This model is trained to distinguish between real mammogram images and their synthetic counterparts. The training data consists of paired original `.png` images and corresponding `.npz` synthetic images.

### 🔍 How It Works

- Uses a CNN to learn differences between real and synthetic scans.
- Custom generator ensures balanced batches of both types.
- Binary classification:  
  - `1` → Original image  
  - `0` → Synthetic image

### 💡 Use Case

Useful for validating datasets and maintaining data quality by detecting artificially generated images.

---

## 🧬 Classifier 2: Cancerous vs. Non-Cancerous Mammogram Detection

📍 **Notebook Link**: [C_Classifier.ipynb](https://github.com/VidhyaSree-N/classifier_synthetic_original_mammograms/blob/main/C_Classifier.ipynb)

This model detects whether a mammogram shows signs of cancer. It uses `.npz` files containing preprocessed mammographic image arrays.

### 🔍 How It Works

- Data is Z-score normalized to standardize intensity.
- CNN architecture trained on balanced datasets (cancerous and non-cancerous).
- Binary classification:
  - `1` → Cancerous  
  - `0` → Non-cancerous

### 💡 Use Case

Helps in early research and prototyping of assistive tools for radiologists or machine learning-based diagnostic pipelines.

---

## 📊 Model Performance

### Classifier 1: Synthetic vs. Original Classifier

This classifier currently achieves **very high accuracy (~99%)**, which highlights an important insight:

#### 🔬 Observations & Learnings

- The classifier likely benefits from **inherent differences in data formats**:
  - Original images are `.png` files.
  - Synthetic images are `.npz` arrays, visually and structurally distinct.
- The difference is so pronounced that **even a human** can tell them apart easily.
- To reduce this disparity, the following steps were explored:
  - Adjusted contrast and normalization to make real images resemble synthetic ones.
  - Experimented with different architectures, optimizers, batch sizes, and epoch lengths.

Despite these changes, the model maintained high accuracy — suggesting it may be learning format-based cues rather than true semantic differences.

#### 🚧 Future Directions

- Reformat datasets to reduce bias (e.g., use the same file type for both classes).
- Generate more realistic synthetic data (e.g., using GANs).
- Explore domain adaptation techniques to enforce meaningful feature learning.

> ✨ **Contributions Welcome**: This classifier offers a valuable opportunity to explore dataset bias, synthetic data evaluation, and domain shift. Contributions are welcome to improve fairness, realism, and robustness!

### Classifier 2: Cancer Classifier
- Achieves ~77% accuracy on validation sets with dropout and normalization strategies.

> ⚠️ These models are intended for **research and development only** and are **not validated for clinical use**.

---

## 🧰 Tech Stack

| Technology         | Purpose                                     |
|--------------------|---------------------------------------------|
| **Python**         | Programming language                        |
| **TensorFlow / Keras** | Deep learning frameworks               |
| **NumPy**          | Numerical computing                         |
| **OpenCV**         | Image preprocessing                         |
| **scikit-learn**   | Data splitting                              |
| **Matplotlib**     | Visualization                               |
| **ImageDataGenerator** | Data augmentation                     |

---

## 📄 License

This project is licensed under the **MIT License**.

---
