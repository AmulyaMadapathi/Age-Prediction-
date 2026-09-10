# Age Prediction Using Machine Learning

## 📌 Overview

This project is a Python-based web application that analyzes facial images to detect faces and predict age and gender using pre-trained deep learning models with OpenCV.

The application provides a simple Flask-based interface where users can upload an image and receive age and gender predictions along with application-specific insights.

## ✨ Features

- Face detection from uploaded facial images
- Age prediction using predefined age ranges
- Gender prediction
- Flask-based web interface
- Image upload support
- Image URL processing support
- Age-based insights for:
  - Health
  - Security
  - Marketing
- Displays the detected face and predicted age on the image

## 🧠 How It Works

The application follows these steps:

1. The user uploads an image or provides an image URL.
2. OpenCV's face detection model identifies faces in the image.
3. The detected face is extracted with padding.
4. The extracted face is processed using the age and gender models.
5. The application predicts the age range and gender.
6. Based on the selected application type, additional insights are generated.
7. The processed image and prediction results are returned to the user.

## 🛠️ Technologies Used

- Python
- Flask
- OpenCV
- NumPy
- Requests
- TensorFlow / Keras (for the separate model-training component)

## 🤖 Models Used

The application uses OpenCV DNN models stored in the `models/` directory:

- Face detection model
- Age prediction model
- Gender prediction model

The age prediction model classifies faces into the following age ranges:

```text
(0-2)
(4-6)
(8-12)
(15-20)
(22-29)
(35-43)
(48-53)
(60-100)