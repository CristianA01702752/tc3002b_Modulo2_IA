#Librerías
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#variables globales
train_path = 'train'
augmented_path = 'augmented'
width, height = 150, 150
batch_size = 64

#Preprocesamiento del conjunto de entrenamiento
train_datagen = ImageDataGenerator(
							rescale = 1./255,
							rotation_range = 90,
							width_shift_range = 0.2,
							height_shift_range = 0.2,
							shear_range = 0.2,
							zoom_range = 0.2,
							horizontal_flip = True,)

train_generator = train_datagen.flow_from_directory(
							train_path,
							target_size = (width, height),
							batch_size = batch_size,
							class_mode ='categorical',
                        	save_to_dir= augmented_path,
              				save_prefix='aug',
              				save_format='png')
#Etiquetas de las clases
classes = {
    0: 'Apple Pie',
    1: 'Baked Potato',
    2: 'Cheesecake',
    3: 'Chicken Curry',
    4: 'Crispy Chicken',
    5: 'Donut',
    6: 'Fries',
    7: 'Hot Dog',
    8: 'Ice Cream',
    9: 'Omelette',
    10: 'Sandwich',
    11: 'Sushi',
    12: 'Taco',
}
#Impresión de una muestra de las imágenes generadas en el preprocesamiento
images, labels = train_generator[0]
num_imgs = min(images.shape[0], 20)
fig, axarr = plt.subplots(2, 10, figsize=(5, 5))
for i in range(num_imgs):
    row = i // 10
    col = i % 10
    axarr[row, col].imshow(images[i])
    label_index = np.argmax(labels[i])
    axarr[row, col].set_title(classes[label_index])
    axarr[row, col].axis('off')
plt.tight_layout()
plt.show()