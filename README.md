# Waste Classification Project

This project provides a complete pipeline for training and running real-time waste classification using **ResNet50** and **VGG16** models, combined with **YOLO** for object detection.

---

## Requirements

* **Python version**: 3.9 – 3.11 (recommended)

  * ⚠️ Python 3.12 or newer is not yet fully compatible with GPU training.
* **pip**: latest version
* **GPU support** (optional but recommended):

  * NVIDIA CUDA/cuDNN (Windows & Ubuntu)
  * Apple Metal (macOS)

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd <your_repo_folder>
```

### 2. Create a Virtual Environment

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Ubuntu / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Training the Models

You can train the classification models either locally or in Google Colab.

### Local Training

1. Start Jupyter Lab:

   ```bash
   jupyter lab
   ```
2. Open the provided `.ipynb` notebook file.
3. Run all cells to train the models.

This will generate two trained models saved in your project directory:

* `resnet50_garbage_classifier_model.keras`
* `vgg16_garbage_classifier_model.keras`

---

## Real-Time Detection

Once models are trained, you can run real-time detection using YOLO + classifier:

```bash
# Activate environment first
# Ubuntu / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Run detection script
python detection.py
```

### How It Works

1. **YOLO** detects objects and draws bounding boxes (bbox).
2. Each detected object is passed to the trained classifier (ResNet50 / VGG16).
3. The classifier predicts the waste category in **real time**.

---

✅ With this setup, you can train, evaluate, and run real-time waste detection and classification seamlessly on Windows, Ubuntu, or macOS.
