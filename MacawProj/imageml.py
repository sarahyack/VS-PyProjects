"""
    First Iteration: Didn't quite understand that the images had to be all the same size and that color needed to be taken into account.
    Second Iteration: With GPT's help, I re-wrote the model to include Convolutional layers so that I could take everything into account.
    """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import pickle

# Load image data
with open("C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj\\image_data.pkl", "rb") as f:
    image_data = pickle.load(f)

print(type(image_data))  # Check the type of image_data

# If image_data is a tuple, unpack it like this:
if type(image_data) == tuple:
    X, y = image_data
else:
    X = np.array(image_data['images'])
    y = np.array(image_data['labels'])

# Add debugging lines here
print("Shape of X:", np.shape(X))
print("Shape of y:", np.shape(y))

# Reshape the images to have a single color channel and resize them
# X_reshaped = tf.image.resize(X, [128, 128])
X = np.expand_dims(X, axis=-1)


# More debugging lines
""" print("Shape of X:", X.shape)
print("Shape of X_reshaped:", X_reshaped.shape)
print("Length of X_reshaped:", len(X_reshaped))
print("Length of y:", len(y))
print("First 5 of X_reshaped:", X_reshaped[:5])
print("First 5 of y:", y[:5]) """

# Split the data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3)
X_test, X_cv, y_test, y_cv = train_test_split(X_, y_, test_size=0.5)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(3, activation='softmax', kernel_regularizer=l2(0.01))
], name='my_conv_model')

# Compile the model
model.compile(loss=SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam())

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=200,
          validation_data=(X_cv, y_cv))

model.save("C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj\\my_model")

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# To load a saved model (you'd typically do this in a new session)
# Uncomment the following line when you need it:
# loaded_model = tf.keras.models.load_model("C:\\Users\\Sarah\\Documents\\My Projects\\Coding\\VS PyProjects\\MacawProj\\my_model")

# If you load the model, you would use `loaded_model` in place of `model` below.

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
num_errors = np.sum(y_pred != y_test)

print(model.summary())

print("First five predicted labels: ", y_pred[:5])
print("First five true labels: ", y_test[:5])
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(num_errors)