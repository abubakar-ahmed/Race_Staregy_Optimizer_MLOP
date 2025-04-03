import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_data):
    """
    Preprocesses input data for the tire compound prediction model.

    Args:
        input_data (dict): A dictionary containing the input features.

    Returns:
        numpy.ndarray: The preprocessed input data ready for prediction.
    """

    # 1. Convert 'Weather' and 'Driving_Style' to numerical values
    weather_mapping = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Mixed": 3}
    driving_style_mapping = {"Aggressive": 0, "Balanced": 1, "Conservative": 2}

    input_data["Weather"] = weather_mapping[input_data["Weather"]]
    input_data["Driving_Style"] = driving_style_mapping[input_data["Driving_Style"]]

    # 2. Create a DataFrame from the input_data dictionary
    input_data_df = pd.DataFrame([input_data])

    # 3. Select the features used during training
    features = ['Lap', 'Temperature', 'Weather', 'Driving_Style', 'Laps_Remaining']
    input_data_processed = input_data_df[features]

    # 4. Scale the input data using the loaded scaler
    scaler = joblib.load("scaler.pkl")  # Load the saved scaler
    scaled_input_data = scaler.transform(input_data_processed)

    return scaled_input_data