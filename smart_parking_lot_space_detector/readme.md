# 🚘 Smart Parking Space Detector

## 🧠 Overview
The **Smart Parking Space Detector** is a simple machine learning–based application that helps drivers and parking managers quickly identify available parking spaces.  
It uses a trained **Support Vector Machine (SVC)** classifier to analyze images of parking slots and determine whether each space is **Empty** ✅ or **Not Empty** 🚗.  

The model was trained on images resized to **15×15 pixels**, flattened, and classified using a **GridSearchCV-optimized SVC** model.  
The user interface was built using **Gradio** for an intuitive and interactive experience.

---

## ⚙️ Features
- 🖼️ Upload an image of a parking space  
- 🤖 Predicts whether the space is **Empty** or **Not Empty**  
- 🎨 Displays the result visually on the image with color-coded labels  
- 🌐 Interactive web interface built with **Gradio**

---

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:**  
  - `scikit-learn` – Model training and prediction  
  - `Gradio` – Web interface  
  - `NumPy`, `scikit-image`, `Pillow` – Image preprocessing  
  - `pickle` – Model serialization

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/MarilynMaika/smart-parking-lot-space-detector.git
cd smart-parking-space-detector
