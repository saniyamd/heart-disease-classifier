# Heart Disease Classifier

This project uses deep learning models to classify various types of heart disease based on structured medical data.

### Conditions Predicted:
- No Heart Disease
- Angina
- Myocardial Infarction
- Arrhythmia


## Models

Two pre-trained convolutional neural networks were adapted for this task:

- **VGG16**: Input reshaped and resized to 32×32
- **InceptionV3**: Input reshaped and resized to 75×75

Although these models are built for image data, the clinical features were transformed into image-like structures to enable compatibility.



## Dataset

- File: `cleaned_merged_heart_dataset.csv`
- Contains patient health records with features such as:
  - Age, sex, resting blood pressure, cholesterol
  - Fasting blood sugar, ECG results, maximum heart rate, and more
- Target values were programmatically reclassified into four categories based on feature thresholds



## How to Run

Install the required packages (`tensorflow`, `pandas`, `numpy`, etc.) and execute one of the following:

```bash
python vgg16_model.py
