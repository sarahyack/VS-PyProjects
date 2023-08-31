"""
	Early Machine Learning Project
		Iris Classification Project:
			First iteration:
				80/20 data split, with the cv/test data divided in half in the 20%
				Three Dense Layers, learning rate at 0.009, and various loss spikes.
				Fiddled with it for a while, and finally moved on to second iteration.
			Second iteration:
				Five Dense layers, and a Dropout layer. Learning rate was still alternating between 0.008 and 0.009,
				but losses kept spiking and not spiking, or the model would perform much better on the CV set than the training set.
				Changed the data structure to 70/30, making the CV/Test 15/15. Model performed much more erratically (in regards to loss)
				Note: I hadn't implemented the model.predict in any way yet, and was just focused on the losses.
				Finally decided I was making it too complicated. Moved on to third and current iteration.
			Third iteration:
				Three Dense Layers (once again)
				70/30 data split, cv/test 15/15
				Got rid of the specific learning rate, used Adam optimizer's default.
				Added model.predict functionality. Tried experimenting with the previous iterations, seemed to perform just fine.
				Added count of errors.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.activations import linear, relu, sigmoid, softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
with open(r"C:\Users\Sarah\Documents\My Projects\Coding\ML Testing\Iris Solo Project\iris\iris.data", "r") as file:
	data = pd.read_csv(file, header=None)

data.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']

data.head()
data.describe()
sns.pairplot(data)

le = LabelEncoder()
X = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = le.fit_transform(data['Class'])
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3)
X_test, X_cv, y_test, y_cv = train_test_split(X_, y_, test_size=0.5)

# Kept this for future reference
# print(f"the shape of the training set (input) is: {X_train.shape}")
# print(f"the shape of the training set (target) is: {y_train.shape}\n")
# print(f"the shape of the cross validation set (input) is: {X_cv.shape}")
# print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
# print(f"the shape of the test set (input) is: {X_test.shape}")
# print(f"the shape of the test set (target) is: {y_test.shape}")

model = Sequential([
	# Dense(50, activation='relu'),
	# Dropout(0.2),
	# Dense(35, activation='relu'),
	# Dense(25, activation='relu'),
	Dense(15, activation='relu'),
	Dense(10, activation='relu'),
	Dense(3, activation='softmax')
], name = 'my_model')

model.compile(loss = SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.Adam())
#Got rid of learning_rate=0.009

history = model.fit(
	X_train, y_train,
	epochs=200,
	validation_data=(X_cv, y_cv)
)
print("Done!\n")

#Intial plotting of cost
# plt.figure(figsize=(15, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
num_errors = np.sum(y_pred != y_test)

print(model.summary())

print("First five predicted labels: ", y_pred[:5])
print("First five true labels: ", y_test[:5])
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(num_errors)