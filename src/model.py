import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_processed_data(file_path="processed_data.npz"):
    """
    Load preprocessed data from an .npz file.

    Args:
        file_path (str): Path to the processed data file.

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays): Train-test split data.
    """
    data = np.load(file_path)
    X, y = data["X"], data["y"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_best_model(model_path="/content/forth_model.keras"):
    """
    Load the best-performing model.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        model (tf.keras.Model): Loaded model.
    """
    return tf.keras.models.load_model(model_path)

def retrain_model(model, X_train, y_train, epochs=100, batch_size=32):
    """
    Retrain the existing model on new data.

    Args:
        model (tf.keras.Model): The pre-trained model.
        X_train (numpy array): Training feature matrix.
        y_train (numpy array): Training labels.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        model (tf.keras.Model): The retrained model.
    """
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return model

def save_model(model, model_path="models/best_model.keras"):
    """
    Save the trained model.

    Args:
        model (tf.keras.Model): The trained model.
        model_path (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved at: {model_path}")

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_best_model()
    retrained_model = retrain_model(model, X_train, y_train)
    save_model(retrained_model)
