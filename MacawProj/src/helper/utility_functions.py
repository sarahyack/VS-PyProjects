import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def load_data(file_path):
    with open(file_path, "rb") as f:
        image_data = pickle.load(f)
    if type(image_data) == tuple:
        X, y = image_data
    else:
        X = np.array(image_data['images'])
        y = np.array(image_data['labels'])
    return X, y

def split_dataset(X, y, test_size, test_split):
    """
    Split the dataset into training, testing, and cross-validation sets.

    Parameters:
        X (array-like): The input features.
        y (array-like): The target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        test_split (float): The proportion of the training set to include in the cross-validation split.

    Returns:
        tuple: A tuple containing the training, testing, and cross-validation sets.
            - X_train (array-like): The training set features.
            - X_test (array-like): The testing set features.
            - X_cv (array-like): The cross-validation set features.
            - y_train (array-like): The training set target variable.
            - y_test (array-like): The testing set target variable.
            - y_cv (array-like): The cross-validation set target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=test_split)
    return X_train, X_test, X_cv, y_train, y_test, y_cv

def preprocess_images(X_train):
    X = np.expand_dims(X_train, axis=-1)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    return datagen

def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model
    else:
        print(f"Model not found at {path}")
        return None

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def print_model_summary(model):
    print("Model Summary:")
    model.summary()

def plot_metrics(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype("int")
    acc = accuracy_score(y_test, predicted_classes) * 100
    correct_count = (predicted_classes == y_test).sum()
    incorrect_count = (predicted_classes != y_test).sum()
    print(f"Accuracy: {acc}%")
    print(f"Correct Count: {correct_count}")
    print(f"Incorrect Count: {incorrect_count}")
    return acc, correct_count, incorrect_count

