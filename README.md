# Real-Time Face Recognition with Python, OpenCV, and MTCNN

## 📌 Function Description

This project implements a real-time face recognition system using webcam input. It detects faces using MTCNN, extracts deep facial embeddings via FaceNet, and compares them against a set of known faces to identify individuals. The system is built for real-time performance and ease of deployment, making it ideal for demonstration, research, and educational purposes.

---

## 🧠 What the Code Does

- **Registers known faces** from images in a folder (`known_faces/`)
- **Detects faces** in real-time webcam feed using MTCNN
- **Extracts facial embeddings** using FaceNet
- **Compares embeddings** with registered ones using Euclidean distance
- **Displays bounding boxes** and names of recognized individuals on video stream

---

## 🔧 Technologies Used

- **Python**
- **OpenCV** – for video capture and image processing
- **MTCNN** – for accurate face detection
- **keras-facenet** – for generating facial embeddings
- **NumPy & SciPy** – for numerical operations and similarity calculations

---

## ⚙️ Setup Instructions

### 1. Clone the repository (or download the code)

```
   git clone https://github.com/Vaithishworks/Face_Recognition.git
   cd Face_Recognition
```
### 2. Install the dependencies
```
    pip install numpy==1.24.3 tensorflow==2.11 mtcnn keras-facenet opencv-python scipy
```
### 2. create a folder known_faces and put the photos of faces to be recognised.
```
    mkdir known_faces
```
### 2. Install the dependencies
```
python3 face_rec.py
```

