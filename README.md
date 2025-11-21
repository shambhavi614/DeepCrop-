# 🌱 **DeepCrop: A Deep Learning Framework for Multi-Plant Disease Detection**

DeepCrop is an AI-powered plant disease detection system built using Convolutional Neural Networks (CNNs).
It identifies multiple crop diseases using images and helps farmers take early action to protect crop health.

---

## 📌 **Features**

* 🌿 Detects diseases across multiple plant species
* 🧠 Uses a custom CNN deep learning model
* 📊 90%+ accuracy (depending on training configuration)
* 🗂 Trained on **PlantVillage** dataset
* 🖥 Simple and clean UI for uploading leaf images
* 📁 Well-organized dataset and training pipeline
* 🔍 Real-time prediction support

---

## 📂 **Project Structure**

```
DeepCrop/
│── dataset/
│   ├── train/
│   ├── test/
│   └── validation/
│
│── model/
│   ├── deepcrop_model.h5
│   └── model_training.ipynb
│
│── src/
│   ├── model.py
│   ├── predict.py
│   └── utils.py
│
│── webapp/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── uploads/
│
│── README.md
│── requirements.txt
└── LICENSE
```

---

## 🔧 **Tech Stack**

* **Python**
* **TensorFlow / Keras**
* **NumPy, Pandas, Matplotlib**
* **Flask (for Web App)**
* **PlantVillage Dataset**

---

## 🧠 **Model Architecture (CNN)**

Your CNN includes:

* Convolution layers
* MaxPooling layers
* Flatten layer
* Dense layers
* Softmax output layer

```
Input → Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output
```

---

## 📈 **Training**

To train the model:

```python
python train.py
```

or open Jupyter Notebook:

```bash
jupyter notebook model_training.ipynb
```

---

## 🚀 **Run the Web App**

```bash
python app.py
```

Then open browser:

```
http://localhost:5000
```

---

## 🖼 **Prediction**

Upload a leaf image → Model predicts:

* Healthy / Unhealthy
* Name of disease
* Confidence score

---

## 📥 **Dataset**

This project uses the **PlantVillage** dataset containing 50,000+ labelled images.

Download link (official):
[https://data.mendeley.com/datasets/tywbtsjrjv/1](https://data.mendeley.com/datasets/tywbtsjrjv/1)

---

## 📃 **How to Train the Model? (Short Explanation)**

1. Load dataset
2. Preprocess images (resize, normalization)
3. Split into train/test
4. Build CNN architecture
5. Compile model with Adam optimizer
6. Train 20–30 epochs
7. Save model

---

## 🧪 **Results**

* Training Accuracy: ~95% (varies by crop)
* Test Accuracy: ~90%
* Low loss and stable validation curve

---

## 💡 **Future Improvements**

* Add more crop classes
* Implement MobileNet for faster predictions
* Deploy on mobile app
* Add voice-based explanation system

---

## 📝 **License**

This project is open-source under the MIT License.

---

## 🙌 **Author**

**Shambhavi Jha**
B.Tech (CSE-AI)

---

<img width="1920" height="910" alt="image" src="https://github.com/user-attachments/assets/8718821d-8f83-4831-b24c-56ec832f1f85" />
<img width="1920" height="930" alt="image" src="https://github.com/user-attachments/assets/943c71c0-1176-46b1-b724-3879c450e3fe" />
<img width="1920" height="922" alt="image" src="https://github.com/user-attachments/assets/4f96a65c-50a1-4d72-8883-e338728dfc7c" />
<img width="1920" height="900" alt="image" src="https://github.com/user-attachments/assets/8b322785-7e00-4ce5-bf41-4f47ab96785b" />
<img width="1920" height="916" alt="image" src="https://github.com/user-attachments/assets/8739a2c6-4c6a-454a-887f-ab90fe93ff98" />
<img width="1920" height="914" alt="image" src="https://github.com/user-attachments/assets/5f80978a-1f8a-4be7-8298-42d0edae2746" />
<img width="1920" height="920" alt="image" src="https://github.com/user-attachments/assets/a015f367-600c-46b9-8720-72eb6d4ff9f4" />
<img width="1920" height="928" alt="image" src="https://github.com/user-attachments/assets/21d0ba8e-1619-4141-81f6-8c4a9a8439ad" />







