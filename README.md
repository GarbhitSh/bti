# BTI: Brain Tumor Detection, Classification, and Diagnosis System

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Running the Classifier](#running-the-classifier)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
BTI is a high-accuracy (99.3%) brain tumor detection, classification, and diagnosis system using state-of-the-art deep learning methods. This project leverages powerful neural networks to analyze MRI scans and predict the presence and type of brain tumors, assisting in timely diagnosis and treatment planning.

The dataset used contains 3064 T1-weighted contrast-enhanced images from 233 patients with three kinds of brain tumors:
- **Meningioma**: 708 slices
- **Glioma**: 1426 slices
- **Pituitary Tumor**: 930 slices

## Features
- **High Accuracy**: Achieves 99.3% accuracy in brain tumor detection and classification.
- **Deep Learning**: Utilizes advanced deep learning techniques for image analysis.
- **Comprehensive Diagnosis**: Provides detailed classification of tumor types for better diagnosis.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/garbhitsh/bti.git
    ```
2. Navigate to the project directory:
    ```bash
    cd bti
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Load and preprocess your dataset using the provided scripts.
    ```bash
    python preprocess.py
    ```
2. Train the model on your dataset:
    ```bash
    python train.py
    ```
3. Evaluate the model performance:
    ```bash
    python evaluate.py
    ```
4. Predict using the trained model:
    ```bash
    python predict.py --image_path [path_to_image]
    ```

## Modules
- **[brain_tumor_dataset_preparation.ipynb](brain_tumor_dataset_preparation.ipynb)** - An IPython notebook that contains preparation and preprocessing of the dataset for training, validation, and testing.
- **[torch_brain_tumor_classifier.ipynb](torch_brain_tumor_classifier.ipynb)** - An IPython notebook that contains all the steps, processes, and results of training, validating, and testing our brain tumor classifier.
- **[test.py](test.py)** - A Python script that accepts a path to an image as input, which then classifies the image into one of the three classes.
- **[deploy.py](deploy.py)** - A Python script integrated with a Flask server that starts the web interface on a local server where users can upload MRI images of the brain and get classification results.

**Note:** We have included a few images for testing under the [test_images](test_images) directory.

## Running the Classifier

1. Download the classifier model `.pt` file from this [drive link](https://drive.google.com/file/d/1-rIrzzqpsSg80QG175hjEPv9ilnSHmqK/view?usp=sharing) and place it in a folder named `models` in the same directory where the files of this repository are present.

2. Before running the programs, kindly install the requirements as given in the [Installation](#installation) section of this README.

3. Use the [test.py](test.py) script to classify an image via the terminal:
    ```bash
    python test.py
    ```

4. Use [deploy.py](deploy.py) to access the classifier as an interactive web interface:
    ```bash
    python deploy.py
    ```

## Technologies
- **Numpy** - For linear algebra operations
- **Torch** - PyTorch Deep Learning Framework
- **OS** - To use Operating System methods
- **Random** - To set random seed for reproducibility
- **Pandas** - For DataFrame creation and CSV handling
- **Time** - For date-time operations
- **Seaborn** - For sophisticated visualizations
- **Pickle** - For saving and loading binary files of training data
- **Scikit-Learn** - For evaluating the classifier and cross-validation split
- **Matplotlib** - For visualizing images, losses, and accuracy
- **Google Colab Drive** - For storage and loading operations on Google Colab

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

1. Fork the project.
2. Create your feature branch:
    ```bash
    git checkout -b feature/YourFeatureName
    ```
3. Commit your changes:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature/YourFeatureName
    ```
5. Open a pull request.

## License
This project is licensed under the MIT License
