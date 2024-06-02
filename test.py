import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns

test_path = 'test'

# Image dimensions
width, height = 150, 150
batch_size = 16

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  
)

model = load_model('Food_Classification.keras')
# Predicciones y matriz de confusión

test_generator.reset()
predicted_labels = []
original_labels = []

# Iterate over the test generator to make predictions and store original labels
for i in range(len(test_generator)):
    batch = test_generator[i]
    images, labels = batch
    predictions = model.predict(images)
    
    # Convert predictions to labels
    batch_predicted_labels = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    
    # Append predicted labels to the array
    predicted_labels.extend(batch_predicted_labels)
    
    # Append original labels to the array
    original_labels.extend(labels)

# Convert lists to numpy arrays
predicted_labels = np.array(predicted_labels)
original_labels = np.array(original_labels)


cm = confusion_matrix(original_labels, predicted_labels)

print(classification_report(original_labels, predicted_labels, target_names = list(test_generator.class_indices.keys())))
print(cm)

# Create a DataFrame with the confusion matrix
cm_df = pd.DataFrame(cm, index=original_labels, columns=predicted_labels)

print("Confusion Matrix with Labels:")
print(cm_df)