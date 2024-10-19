# sign_language_model.py*

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the ImageDataGenerator to load images from directories*

train_datagen = ImageDataGenerator(

rescale=1./255,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True

)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the dataset (make sure the 'dataset' folder is structured as described)*

train_set = train_datagen.flow_from_directory(

'dataset/train',

target_size=(64, 64),

color_mode='grayscale',  *# Assuming the images are in grayscale*

batch_size=32,

class_mode='categorical'

)

test_set = test_datagen.flow_from_directory(

'dataset/test',

target_size=(64, 64),

color_mode='grayscale',

batch_size=32,

class_mode='categorical'

)

# Build the CNN model as before*

model = tf.keras.models.Sequential([

tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(26, activation='softmax')  *# 26 classes (A-Z)*

])

# Compile the model*

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model*

model.fit(train_set, epochs=20, validation_data=test_set)

# Save the trained model*

model.save('sign_language_model.h5')