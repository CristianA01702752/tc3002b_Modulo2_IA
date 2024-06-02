import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Lambda, Concatenate, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Paths
train_path = 'train'
validation_path = 'validation'
test_path = 'test'

# Image dimensions
width, height = 150, 150
batch_size = 16

# Image Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    validation_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  
)

# Base model
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(width, height, 3))
conv_base.trainable = True  # Make sure the base model is trainable

# Model structure
input_layer = layers.Input(shape=(width, height, 3))
x = conv_base(input_layer)

# Global average pooling
global_avg_pooling = layers.GlobalAveragePooling2D()(x)

# Flatten layer
flatten_layer = layers.Flatten()(global_avg_pooling)

# Fully connected layers
dense_layer1 = layers.Dense(512, activation='relu')(flatten_layer)
dense_layer2 = layers.Dense(256, activation='relu')(dense_layer1)
output_layer = layers.Dense(13, activation='softmax')(dense_layer2) 

# Create the model
model = models.Model(inputs=input_layer, outputs=output_layer)

model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.00001),
    metrics=['acc']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=60,
    validation_data=val_generator,
    validation_steps=30
)

# Plot training and validation accuracy and loss
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.plot(epochs, train_acc, 'tab:red', label='train accuracy')
plt.plot(epochs, val_acc, 'tab:green', label='validation accuracy')
plt.title('train acc vs val acc')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'tab:red', label='train loss')
plt.plot(epochs, val_loss, 'tab:green', label='validation loss')
plt.title('train loss vs val loss')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('\ntest acc :\n', test_acc)

# Reset the test generator and predict classes
test_generator.reset()
Y_pred = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)

# Classification Report
target_names = list(test_generator.class_indices.keys())
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

model.save('Food_Classification_Improved.h5')
model.save('Food_Classification_Improved.keras')