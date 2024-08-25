# Plant Disease Prediction with CNN and Deep Learning

This project implements a web-based application to identify plant diseases from images of leaves using a Convolutional Neural Network (CNN). The application is built with Streamlit for the front end, allowing users to upload leaf images, predict diseases, and receive detailed information on the predicted diseases, including control measures.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Run the Application](#run-the-application)
- [Docker Setup](#docker-setup)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
- [Usage](#usage)
- [Model Training](#model-training)
- [Configuration Files](#configuration-files)
- [Model](#model)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Project Overview

The Plant Disease Prediction application is designed to help farmers and agricultural professionals identify diseases affecting plants through image analysis. By simply uploading an image of a plant leaf, the app uses a deep learning model to classify the disease and provide relevant information and control measures.

## Features

- **Image Classification**: Upload a plant leaf image to classify its disease using a pre-trained CNN model.
- **Disease Information**: Obtain detailed information about the predicted disease.
- **Control Measures**: Quickly search for ways to control the detected disease via a Google search link.
- **User-Friendly Interface**: Built with Streamlit, the application offers a clean and easy-to-use interface.

## Project Structure

```
Plant Disease Prediction with CNN Deep Learning/
│
├── app/
│   ├── Dockerfile
│   ├── class_indices.json
│   ├── config.toml
│   ├── credentials.toml
│   ├── disease_info.json
│   ├── main.py
│   ├── requirements.txt
│   └── trained_model/
│       └── plant_disease_prediction_model.h5
│
├── model_training_notebook/
│   └── train_model.ipynb
│
└── test_images/
    └── (sample images for testing)
```

- **app/**: Contains the main application files including the Dockerfile, configuration files, and the trained model.
- **model_training_notebook/**: Includes the Jupyter notebook used to train the model.
- **test_images/**: Directory for storing sample images to test the application.

## Installation

To run this project locally, follow the steps below:

### Clone the Repository

```bash
git clone https://github.com/Pahinithi/Plant-Disease-Prediction-with-CNN-Deep-Learning
cd Plant-Disease-Prediction/app
```

### Install Dependencies

Ensure you have Python 3.10 installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Run the Application

Run the application using Streamlit:

```bash
streamlit run main.py
```

The application will be accessible at `http://localhost:8501`.

## Docker Setup

You can also run the application inside a Docker container for easy deployment.

### Build the Docker Image

```bash
docker build -t plant-disease-prediction .
```

### Run the Docker Container

```bash
docker run -p 8501:80 plant-disease-prediction
```

The application will be accessible at `http://localhost:8501`.

## Usage

1. **Upload Image**: Click the "Upload an image..." button to upload an image of a plant leaf.
2. **Classify Disease**: Once the image is uploaded, click "Classify" to predict the disease.
3. **View Results**: The application will display the predicted disease along with information and control measures.

## Model
- Link : https://dms.uom.lk/s/W3SGjkgjKBpA3cb
- Video Link : https://drive.google.com/file/d/1FSig83tqDVxLt8VH0KXcaE51puVnfled/view?usp=sharing

  <img width="1728" alt="DL08" src="https://github.com/user-attachments/assets/19d1012a-dc8d-45c7-a7ad-c7368ca7ccb1">

  

## Model Training

The CNN model was trained using a dataset of plant leaf images. The training was conducted using the `train_model.ipynb` notebook, which is available in the `model_training_notebook` directory. The final model is saved as `plant_disease_prediction_model.h5` in the `trained_model` directory.

### Model Architecture
- **Input Layer**: 224x224x3 (RGB images resized to 224x224 pixels)
- **Convolutional Layers**: Several layers with ReLU activation and max-pooling.
- **Fully Connected Layers**: Dense layers leading to the output layer.
- **Output Layer**: Softmax activation function with multiple output classes representing different plant diseases.

## Configuration Files

- **config.toml**: Contains Streamlit configuration settings.
- **credentials.toml**: Stores Streamlit credentials.
- **class_indices.json**: Maps class indices to disease names.
- **disease_info.json**: Stores detailed information about each disease.

### Streamlit Configuration

```toml
[global]
showWarningOnDirectExecution = false

[logger]
level = "debug"

[server]
headless = true
port = 80
enableCORS = true

[browser]
serverAddress = "0.0.0.0"
gatherUsageStats = true
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- **TensorFlow**: For providing the framework to develop the CNN model.
- **Streamlit**: For the interactive and user-friendly web interface.
- **Pillow**: For image processing capabilities.
- **Docker**: For containerizing the application for easy deployment.

## Contact

For any questions or suggestions, feel free to contact me:

**Nithilan**
- GitHub: [Pahinithi](https://github.com/Pahinithi)
- Email: [nithilan32@gmail.com](mailto:nithilan32@gmail.com)
