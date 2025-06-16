import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv('cleaned_merged_heart_dataset.csv', names=column_names, header=0)

heart_disease_types = {
    0: "No Heart Disease",
    1: "Angina",
    2: "Myocardial Infarction",
    3: "Arrhythmia"
}

def categorize_disease(row):
    if row['target'] == 1:
        if row['chol'] > 250 or row['trestbps'] > 150 or row['oldpeak'] > 3.0:
            return 2 
        elif row['restecg'] == 2 or row['thalachh'] < 100 or row['slope'] == 2 or row['exang'] == 1 or row['thal'] > 2:
            return 3 
        else:
            return 1  
    return 0

data['target'] = data.apply(categorize_disease, axis=1)

X = data.drop('target', axis=1).values 
y = data['target'].values 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
def reshape_to_2d(X):
    n_samples, n_features = X.shape
    new_side = int(np.ceil(np.sqrt(n_features)))
    total_cells = new_side * new_side
    X_padded = np.zeros((n_samples, total_cells))
    X_padded[:, :n_features] = X
    return X_padded.reshape((n_samples, new_side, new_side, 1))

X_2d = reshape_to_2d(X_scaled)
X_3ch = np.repeat(X_2d, 3, axis=-1)

X_resized = tf.image.resize(X_3ch, (32,32)).numpy()
y = tf.keras.utils.to_categorical(y, num_classes=4) 
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=42)
def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (32, 32, 3)
num_classes = 4 
vgg16_model = create_vgg16_model(input_shape, num_classes)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = vgg16_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    
    plt.show()

plot_training_history(history)
def predict_heart_disease(user_input):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_2d = reshape_to_2d(input_scaled)
    input_3ch = np.repeat(input_2d, 3, axis=-1)
    input_resized = tf.image.resize(input_3ch, (32,32)).numpy()
    prediction = vgg16_model.predict(input_resized)
    predicted_class = np.argmax(prediction)
    
    plt.imshow(input_resized[0])
    plt.axis('off')
    plt.title(heart_disease_types[predicted_class])
    plt.show()
    
    return heart_disease_types[predicted_class]

def visualize_predictions(samples):
    plt.figure(figsize=(12, 4))
    for i, sample in enumerate(samples):
        input_array = np.array(sample).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_2d = reshape_to_2d(input_scaled)
        input_3ch = np.repeat(input_2d, 3, axis=-1)
        input_resized = tf.image.resize(input_3ch, (32, 32)).numpy()
        
        prediction = vgg16_model.predict(input_resized)
        predicted_class = np.argmax(prediction)
        
        plt.subplot(1, len(samples), i + 1)
        plt.imshow(input_resized[0])
        plt.axis('off')
        plt.title(heart_disease_types[predicted_class])
    plt.show()

test_samples = [[65, 1, 2, 160, 280, 1, 1, 120, 0, 4.0, 2, 1, 2],
                [63,1,3,145,233,1,0,150,0,2.3,0,0,1] ,
                 [60, 1, 1, 140, 294, 0, 0, 153, 0, 1.3, 1, 0, 2],
                [60, 1, 2, 140, 260, 0, 0, 140, 1, 2.0, 2, 1, 1],
                [56, 1, 1, 140, 204, 0, 0, 103, 0, 1.3, 1, 0, 2]]

visualize_predictions(test_samples)