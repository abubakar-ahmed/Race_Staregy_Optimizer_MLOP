import tensorflow as tf
import numpy as np

def predict_tire_compound(input_data):
    """
    Predicts the tire compound using the loaded model.

    Args:
        input_data (numpy.ndarray): The preprocessed input data.

    Returns:
        str: The predicted tire compound.
    """

    # 1. Load the trained model
    model = tf.keras.models.load_model("tire_compound_predictor.h5")

    # 2. Make the prediction
    prediction = model.predict(input_data)

    # 3. Get the predicted class label
    class_labels = ["Hard", "Intermediate", "Medium", "Soft", "Wet"]
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class
    predicted_tire_compound = class_labels[predicted_class_index]

    return predicted_tire_compound