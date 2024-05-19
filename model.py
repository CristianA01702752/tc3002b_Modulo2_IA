import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'train'
validation_path = 'validation'
test_path = 'test'

width, height = 150, 150
batch_size = 64
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
							class_mode ='categorical')

val_datagen = ImageDataGenerator(1./255)

val_generator = val_datagen.flow_from_directory(
							validation_path,
							target_size = (width, height),
							batch_size = batch_size,
							class_mode= 'categorical')

test_datagen = ImageDataGenerator(1./255)

test_generator = test_datagen.flow_from_directory(
							test_path,
							target_size = (width, height),
							batch_size = batch_size,
							class_mode= 'categorical')