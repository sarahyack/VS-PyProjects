import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import pickle

# Path to save or load the model
model_path = "C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj\\my_model"

# Load image data
with open("C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj\\image_data.pkl", "rb") as f:
    image_data = pickle.load(f)

print(type(image_data))  # Check the type of image_data

# If image_data is a tuple, unpack it
if type(image_data) == tuple:
    X, y = image_data
else:
    X = np.array(image_data['images'])
    y = np.array(image_data['labels'])

# Debugging lines
print("Shape of X:", np.shape(X))
print("Shape of y:", np.shape(y))

# Reshape the images to have a single color channel
X = np.expand_dims(X, axis=-1)

# Split the data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3)
X_test, X_cv, y_test, y_cv = train_test_split(X_, y_, test_size=0.5)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Check if a saved model exists
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded.")
else:
    # If not, create a new model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(3, activation='softmax', kernel_regularizer=l2(0.01))
    ], name='my_conv_model')
    model.compile(loss=SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam())
    print("New model created.")

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=200,
                    validation_data=(X_cv, y_cv))

# Save the model after training
model.save(model_path)

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
num_errors = np.sum(y_pred != y_test)

print(model.summary())

print("First five predicted labels: ", y_pred[:5])
print("First five true labels: ", y_test[:5])
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(num_errors)
