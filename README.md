# Date Palm White Scale Disease Detection

This project focuses on detecting and categorizing the infestation level of White Scale Disease in date palm trees, a significant issue for Moroccan agriculture. Date palms are essential for both economic and nutritional purposes, especially in Oasis agriculture. However, the White Scale Disease, caused by the *Parlatoria Blanchard* bug, leads to degraded fruit quality and can cause death in trees if the infestation is high, affecting yield and leading to regular losses for farmers. This project aims to empower farmers with precise tools for detecting the disease early, improving production quality and yield.

## Table of Contents
- [Project Overview](#project-overview)
- [Methods](#methods)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Challenges](#challenges)
- [Future Work](#future-work)
- [Usage](#usage)
- [Contributors](#contributors)

## Project Overview
This project utilizes Machine Learning (ML) algorithms to identify and categorize White Scale Disease in date palms. Techniques like Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest, and Light Gradient Boosting Machines (LightGBM) were used to classify images of infected and healthy date palm leaflets. The models were trained on features extracted using Gray Level Co-Occurrence Matrix (GLCM) and Hue, Saturation, and Value (HSV) color space techniques.

An interactive web application was developed to allow users to upload a leaflet image and receive a prediction on the infestation level.

## Methods
1. **Image Preprocessing**: 
   - **Gray Level Co-Occurrence Matrix (GLCM)** and **HSV** were used to extract texture and color features from each image.
2. **Data Augmentation**: 
   - Given the unbalanced nature of the dataset, data augmentation techniques were applied to balance class distribution.
3. **Model Training**: 
   - Algorithms used: SVM, KNN, Random Forest, LightGBM.
   - **Hyperparameter tuning**: GridSearchCV was used for optimal parameter selection.
   - **Cross-Validation**: Ensured the robustness of models.
4. **Web Application**: 
   - A Flask-based user interface was created, allowing users to upload images for real-time disease detection. The trained model is loaded using Joblib for inference.

## Technologies Used
- **Machine Learning Algorithms**: SVM, KNN, Random Forest, LightGBM
- **Image Processing**: Gray Level Co-Occurrence Matrix (GLCM), HSV
- **Backend**: Flask API
- **Frontend**: HTML/CSS (for simple UI)
- **Data Augmentation**
- **Hyperparameter Tuning**: GridSearchCV
- **Cross-Validation**

## Features
- **Image Upload**: Users can upload images of date palm leaflets.
- **Disease Classification**: Provides the infestation level of White Scale Disease.
- **Real-time Prediction**: Fast and efficient predictions on uploaded images.
- **Interactive UI**: Simple and user-friendly interface built with Flask API.

## Challenges
1. **Data Imbalance**: Limited samples for some infestation levels, making it challenging to maintain balanced classes.
2. **Model Compatibility**: Adapting ML module versions to ensure compatibility with Flask and Joblib for smooth model loading.
3. **Limited Dataset**: Working with a small dataset required effective data augmentation to improve model accuracy.

## Future Work
- **Expand Dataset**: Obtain more image samples to improve model accuracy and robustness.
- **Enhance UI**: Improve the interface for better usability.
- **Additional ML Techniques**: Experiment with additional algorithms and ensemble methods for potential accuracy gains.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Deepak636216/Date-Palm-Disease-Detection.git
