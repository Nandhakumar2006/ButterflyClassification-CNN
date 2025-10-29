# 🦋 Butterfly Species Classification using CNN

A deep learning project that classifies butterfly species using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed using Gradio for an interactive web interface.

## 📘 Project Overview

This project aims to accurately classify butterfly images into their respective species. It uses a custom CNN model trained on labeled butterfly images. The model learns distinct visual features such as color patterns, shapes, and textures to distinguish between species.

### 🚀 Features

Custom-built CNN architecture using TensorFlow/Keras

Data augmentation using ImageDataGenerator for better generalization

Real-time image classification through a Gradio web app

Visualization of training accuracy and loss

Modular and well-documented codebase

### 🧠 Model Architecture

The CNN model consists of:

Multiple Conv2D and MaxPooling2D layers for feature extraction

Flatten and Dense layers for classification

Dropout for regularization

Optimized using Adam optimizer

### 🗂️ Dataset

You can organize your dataset as follows:

dataset/
│
├── train/
│   ├── image_1/
│   ├── image_2/
│   └── ...
│
└── test/
    ├── image_3/
    ├── image_4/
    └── ...


Each class folder should contain the respective butterfly images.

### ⚙️ Installation & Setup

Clone the repository and install dependencies:

git clone https://github.com/<your-username>/ButterflyClassification-CNN.git
cd ButterflyClassification-CNN
pip install -r requirements.txt

### ▶️ Usage
1. Train the model

If you want to retrain the model:

python app.py

2. Run the Gradio app

Once trained or if you already have a model file:

python app.py


Then open the link provided by Gradio (e.g., http://127.0.0.1:7860) to access the web interface.

### 🌐 Gradio Interface

Upload or drag & drop an image of a butterfly into the Gradio app.
The model predicts the butterfly species and displays the result instantly.

### Example UI:

+---------------------------------------------+
| Upload Butterfly Image  | Predict Species  |
+---------------------------------------------+
| 🦋 Image preview here                     |
| Prediction: Monarch Butterfly              |
+---------------------------------------------+

### 📊 Training Visualization

During training, accuracy and loss graphs are plotted using matplotlib for better insight into the model’s performance.

### 📦 Dependencies

All dependencies are listed in requirements.txt
:

pandas
numpy
tensorflow
matplotlib
gradio

### 🧑‍💻 Technologies Used

Python 3.10+

TensorFlow / Keras

NumPy & Pandas

Matplotlib

Gradio

### 📈 Future Improvements

Add more butterfly species for better coverage

Fine-tune using transfer learning (e.g., VGG16, ResNet50)

#APP LINK

https://huggingface.co/spaces/nandha-01/ButterflyClassification-CNN
