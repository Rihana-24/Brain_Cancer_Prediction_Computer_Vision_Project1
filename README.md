# 🧠 Brain MRI Tumor Classification & Clinical AI Decision Support

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-95.5%25-brightgreen)
![ROC-AUC](https://img.shields.io/badge/Macro_ROC--AUC-0.994-success)
![License](https://img.shields.io/badge/License-MIT-green)

> **An end-to-end deep learning platform for brain tumor classification using Magnetic Resonance Imaging (MRI).** This project implements and evaluates convolutional neural network architectures using PyTorch and TensorFlow/Keras, with an interactive Streamlit application for model inference, performance evaluation, and visualization.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Clinical Background and Target Classes](#-clinical-background-and-target-classes)
- [Key Features](#-key-features)
- [Dataset Description](#-dataset-description)
- [Deep Learning Architectures](#-deep-learning-architectures)
- [Image Preprocessing](#-image-preprocessing)
- [Model Performance](#-model-performance)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Model Training and Evaluation](#-model-training-and-evaluation)
- [Interactive Streamlit Application](#-interactive-streamlit-application)
- [Future Improvements](#-future-improvements)
- [Clinical Disclaimer](#-clinical-disclaimer)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🩺 Project Overview

Brain tumors represent a major challenge in neurological diagnosis and treatment planning. Magnetic Resonance Imaging (MRI) is widely used to identify and characterize intracranial abnormalities.

However, interpreting medical images requires specialized expertise and can be time-consuming.

This project explores the application of **Deep Learning and Computer Vision** to automate the classification of brain MRI images into four categories:

1. Glioma
2. Meningioma
3. Pituitary Adenoma
4. No Tumor

The project combines two deep learning frameworks:

- **PyTorch:** ResNet-18-based classification model.
- **TensorFlow/Keras:** ResNet-50-based classification model.

Both implementations use transfer learning with pretrained ImageNet weights and custom classification layers.

The resulting models are integrated into an interactive Streamlit application designed to facilitate image classification, model comparison, and performance monitoring.

### Project Highlights

| Component | Description |
|---|---|
| Problem | Brain tumor classification from MRI images |
| Task | Four-class image classification |
| Deep Learning | Convolutional Neural Networks |
| Frameworks | PyTorch and TensorFlow/Keras |
| Architectures | AlexNet, VGG-16, ResNet-18, ResNet-50 |
| Deployment | Streamlit |
| Test Accuracy | 95.5% |
| Macro ROC-AUC | 0.994 |

---

## 🔬 Clinical Background and Target Classes

The classification system identifies four categories of brain MRI images.

| Class | Description | Main Imaging Characteristics |
|---|---|---|
| **Glioma** | Tumor originating from glial cells | Infiltrative appearance, heterogeneous intensity and irregular contours |
| **Meningioma** | Tumor originating from the meninges | Well-defined margins and frequently homogeneous enhancement |
| **Pituitary Adenoma** | Tumor originating from the pituitary gland | Sellar enlargement and possible suprasellar extension |
| **No Tumor** | Normal brain MRI | Preserved anatomical structures without visible tumor-related abnormalities |

These four categories constitute the target labels used during model training and evaluation.

---

## ✨ Key Features

### 1. Dual Deep Learning Implementation

Two independent implementations are provided:

**PyTorch**
- ResNet-18 pretrained backbone.
- Frozen feature extraction layers.
- Custom fully connected classification layer.
- Custom training loop using PyTorch Autograd.
- Adam optimizer.

**TensorFlow/Keras**
- ResNet-50 pretrained backbone.
- Frozen convolutional layers.
- Global Average Pooling.
- Custom Dense classification layer with Softmax activation.
- Custom optimization loop using `tf.GradientTape`.

### 2. Transfer Learning

Pretrained ImageNet architectures are used as feature extractors.

The original classification heads are replaced with custom layers adapted to the four target classes.

This approach reduces the number of trainable parameters and leverages representations learned from large-scale image datasets.

### 3. Consistent Image Normalization

Both frameworks use a common pixel normalization strategy, scaling image intensities to the interval [-1, 1].

### 4. Interactive Web Application

The Streamlit dashboard provides:

- Framework selection between PyTorch and TensorFlow.
- Single-image and batch MRI uploads.
- Predicted tumor category.
- Prediction confidence scores.
- Model evaluation.
- Confusion matrix visualization.
- Classification performance reports.

---

## 📊 Dataset Description

The project uses a benchmark Brain Tumor MRI Dataset aggregated from publicly available Figshare and Kaggle collections.

### Dataset Summary

| Dataset Split | Number of Images | Percentage |
|---|---:|---:|
| Training | 5,712 | 81.3% |
| Testing | 1,311 | 18.7% |
| **Total** | **7,023** | **100%** |

### Class Distribution

| Class | Training | Testing | Total |
|---|---:|---:|---:|
| Glioma | 1,321 | 300 | 1,621 |
| Meningioma | 1,339 | 306 | 1,645 |
| Pituitary Adenoma | 1,457 | 300 | 1,757 |
| No Tumor | 1,595 | 405 | 2,000 |
| **Total** | **5,712** | **1,311** | **7,023** |

The test dataset was maintained separately from model training and hyperparameter selection to provide an independent evaluation of model performance.

---

## 🏛️ Deep Learning Architectures

The project explores the evolution of convolutional neural networks through three influential architectures.

| Architecture | Year | Main Characteristics |
|---|---:|---|
| AlexNet | 2012 | Deep CNN with ReLU activation and max pooling |
| VGG-16 | 2014 | Deep architecture based on stacked 3 × 3 convolutions |
| ResNet | 2015 | Residual connections that facilitate deeper network optimization |

### Why ResNet?

Traditional deep neural networks can experience optimization difficulties as their depth increases.

ResNet addresses this problem through residual learning.

The fundamental residual mapping is:

\[
\mathcal{H}(x)=\mathcal{F}(x)+x
\]

Where:

- \(x\) represents the input.
- \(\mathcal{F}(x)\) represents the residual transformation.
- \(\mathcal{H}(x)\) represents the resulting output.

The identity shortcut helps gradients propagate through deeper network architectures during training.

### Model Configurations

| Component | PyTorch | TensorFlow |
|---|---|---|
| Backbone | ResNet-18 | ResNet-50 |
| Pretrained weights | ImageNet | ImageNet |
| Feature extractor | Frozen | Frozen |
| Classification head | Linear (512 → 4) | Dense (2048 → 4) |
| Output activation | Logits | Softmax |
| Training mechanism | Autograd | GradientTape |

---

## 📐 Image Preprocessing

All MRI images undergo resizing and normalization before being passed to the neural networks.

### Image Resizing

Images are resized to:

\[
224 \times 224
\]

### Pixel Normalization

The normalization formula is:

\[
\hat{x}=\frac{x-127.5}{127.5}
\]

Where \(x\) represents the original pixel intensity in the interval [0, 255].

The normalized image values lie within:

\[
\hat{x}\in[-1,1]
\]

### PyTorch Implementation

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])
```

### TensorFlow Implementation

```python
normalization_layer = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0
)
```

---

## 📈 Model Performance

The classification system was evaluated on an independent test dataset containing 1,311 MRI images.

### Overall Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **95.5%** |
| Macro Precision | 95.5% |
| Macro Recall | 95.3% |
| Macro F1-Score | 95.4% |
| Macro ROC-AUC (OVR) | **0.994** |

### Classification Report

| Class | Precision | Recall | Specificity | F1-Score |
|---|---:|---:|---:|---:|
| Glioma | 0.940 | 0.940 | 0.982 | 0.940 |
| Meningioma | 0.943 | 0.912 | 0.981 | 0.927 |
| No Tumor | 0.966 | 0.990 | 0.985 | 0.978 |
| Pituitary Adenoma | 0.970 | 0.970 | 0.991 | 0.970 |
| **Macro Average** | **0.955** | **0.953** | **0.985** | **0.954** |

### Confusion Matrix

Rows represent actual classes, while columns represent predicted classes.

| Actual / Predicted | Glioma | Meningioma | No Tumor | Pituitary |
|---|---:|---:|---:|---:|
| Glioma | 282 | 11 | 4 | 3 |
| Meningioma | 14 | 279 | 8 | 5 |
| No Tumor | 1 | 2 | 401 | 1 |
| Pituitary | 3 | 4 | 2 | 291 |

### Performance Interpretation

The results indicate strong classification performance across the four categories.

The No Tumor class achieved a recall of 99.0%, indicating that 401 out of 405 normal MRI images were correctly classified.

Meningioma presented the lowest recall (91.2%), suggesting that distinguishing this category from other tumor classes remains an important area for improvement.

---

## 📁 Repository Structure

```text
Brain_Cancer_Prediction_Computer_Vision_Project1/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── Rihanatou_BANKOLE_model.h5
├── Rihanatou_BANKOLE_model.torch
│
├── models/
│   ├── cnn_pytorch.py
│   ├── cnn_tensorflow.py
│   ├── train_pytorch.py
│   └── train_tensorflow.py
│
├── utils/
│   ├── prep_pytorch.py
│   └── prep_tensorflow.py
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── presentation_slides.html
```

### Main Components

| File / Directory | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `main.py` | Main training and evaluation entry point |
| `requirements.txt` | Project dependencies |
| `models/` | Model architectures and training scripts |
| `utils/` | Data preprocessing pipelines |
| `static/` | Application styles and assets |
| `templates/` | Supplementary HTML templates |
| `Rihanatou_BANKOLE_model.h5` | TensorFlow/Keras model |
| `Rihanatou_BANKOLE_model.torch` | PyTorch model checkpoint |
| `presentation_slides.html` | Interactive project presentation |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-cancer-mri-vision.git

cd brain-cancer-mri-vision
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python --version
```

---

## 💻 Model Training and Evaluation

The project provides a unified command-line interface through `main.py`.

### Train the PyTorch Model

```bash
python main.py \
    --framework pytorch \
    --mode train \
    --epochs 10 \
    --lr 0.001 \
    --wd 0.0001 \
    --cuda
```

### Train the TensorFlow Model

```bash
python main.py \
    --framework tensorflow \
    --mode train \
    --epochs 10 \
    --lr 0.001 \
    --wd 0.0001 \
    --cuda
```

### Evaluate the PyTorch Model

```bash
python main.py --framework pytorch --mode eval
```

### Evaluate the TensorFlow Model

```bash
python main.py --framework tensorflow --mode eval
```

### Available Arguments

| Argument | Description | Default |
|---|---|---|
| `--framework` | Select `pytorch` or `tensorflow` | pytorch |
| `--mode` | Select `train` or `eval` | train |
| `--epochs` | Number of training epochs | 10 |
| `--lr` | Adam learning rate | 0.001 |
| `--wd` | Weight decay coefficient | 0.0001 |
| `--cuda` | Enable GPU acceleration when supported | Disabled |

**Note:** GPU acceleration requires compatible hardware, drivers, and framework installations.

---

## 🌐 Interactive Streamlit Application

The project includes an interactive dashboard for MRI image classification and model evaluation.

### Launch the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

### Application Capabilities

1. **Model Selection:** Switch between PyTorch ResNet-18 and TensorFlow ResNet-50.
2. **MRI Image Upload:** Upload individual or multiple MRI scans.
3. **Automated Classification:** Obtain predicted tumor categories.
4. **Confidence Scores:** Visualize model prediction confidence.
5. **Batch Processing:** Process multiple MRI images.
6. **Performance Evaluation:** Generate classification reports and confusion matrices.

---

## 🔭 Future Improvements

Several extensions could improve the robustness, interpretability, and clinical applicability of the system.

### 1. Three-Dimensional CNN Architectures

Explore 3D ResNet architectures to capture spatial relationships across MRI slices rather than treating each slice independently.

### 2. Multimodal MRI Integration

Combine multiple MRI sequences, including:

- T1
- T1-Contrast
- T2
- FLAIR

This could provide richer information for tumor characterization.

### 3. Vision Transformers

Investigate Vision Transformers (ViT) and Swin Transformer architectures to capture long-range spatial dependencies in medical images.

### 4. Explainable Artificial Intelligence

Integrate Grad-CAM and other explainability techniques to visualize the image regions contributing to model predictions.

### 5. Clinical System Integration

Explore DICOM-compatible workflows and Picture Archiving and Communication System (PACS) integration for future research and deployment.

---

## ⚠️ Clinical Disclaimer

This project is developed for **academic, research, and educational purposes**.

The predictions generated by the models are experimental and must not be interpreted as definitive medical diagnoses.

The system has not been established as a clinically validated medical device. It should not replace professional radiological assessment, clinical judgment, or medical advice.

Any clinical application would require additional validation using independent, representative, multi-institutional datasets and appropriate regulatory assessment.

---

## 📄 License

This project is distributed under the MIT License.

See the [LICENSE](LICENSE) file for additional information.

---

## 🤝 Acknowledgments

Special thanks to:

- **Sartaj Bhuvaji and Masoud Nickparvar** for their contributions to publicly available brain tumor MRI datasets.
- The Figshare and Kaggle open-source communities.
- The PyTorch development team.
- The TensorFlow and Keras development teams.
- The Streamlit development community.

---

## 👩‍💻 Author

**Rihanatou BANKOLE**

Biostatistics | Data Science | Artificial Intelligence | Medical Imaging

Interested in applying machine learning and deep learning to biomedical research, healthcare analytics, and clinical decision support.

---

⭐ If you find this project interesting, feel free to star the repository and contribute to its development.
