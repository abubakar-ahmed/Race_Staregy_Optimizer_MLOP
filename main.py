# main.py
from fastapi import FastAPI, File, Request, UploadFile, BackgroundTasks
from typing import Optional
import os
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
import joblib
import shutil
import time
from pymongo import MongoClient


app = FastAPI()


# Load HTML templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Render index.html
@app.get("/home", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Load the trained model
model = tf.keras.models.load_model("models/race_strategy_model.h5")

# Load preprocessing tools (if used)
scaler = joblib.load("models/scaler.pkl")  # Assuming you scaled your data before training

# Define input schema
class TirePredictionInput(BaseModel):
    lap: int
    temperature: float
    weather: str
    driving_style: str
    laps_remaining: int

# Define category mappings (convert categorical inputs to numbers)
weather_mapping = {"Sunny": 3, "Cloudy": 1, "Rainy": 0, "Mixed": 2}
driving_style_mapping = {"Aggressive": 0, "Balanced": 1, "Conservative": 2}

client = MongoClient("mongodb+srv://Abubakar:Captain06@cluster0.fuwd5p4.mongodb.net/")

# Select the database and collection
db = client["RaceOptimizerDB"]
collection = db["RetrainingData"]
@app.post("/predict")
async def predict_tire_compound(input_data: TirePredictionInput):
    try:
        # Convert categorical inputs to numerical values
        weather_num = weather_mapping.get(input_data.weather, -1)
        driving_style_num = driving_style_mapping.get(input_data.driving_style, -1)

        if weather_num == -1 or driving_style_num == -1:
            return {"error": "Invalid categorical value"}

        # Create feature array with ALL 5 features in the correct order
        features = np.array([[
            input_data.lap,          # Feature 1
            input_data.temperature,   # Feature 2
            weather_num,              # Feature 3
            driving_style_num,        # Feature 4
            input_data.laps_remaining # Feature 5
        ]])

        # Scale only the numerical features (lap, temperature, laps_remaining)
        # Assuming your scaler was trained on these 3 numerical features
        numerical_features = features[:, [0, 1, 4]]  # Select lap, temperature, laps_remaining
        numerical_features_scaled = scaler.transform(numerical_features)
        
        # Combine scaled numerical features with categorical features
        features_scaled = np.copy(features)
        features_scaled[:, [0, 1, 4]] = numerical_features_scaled  # Update only numerical features

        # Predict
        prediction = model.predict(features_scaled)
        predicted_class = np.argmax(prediction)

        # Map predicted class back to tire compounds
        tire_classes = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]
        predicted_tire = tire_classes[predicted_class]

        # Business logic rules
        if input_data.weather == "Rainy":
            predicted_tire = "Wet"
        elif input_data.weather != "Rainy" and input_data.laps_remaining <= 20:
            predicted_tire = "Soft"

        # Store prediction in MongoDB
        record = {
            "Lap": input_data.lap,
            "Temperature": input_data.temperature,
            "Weather": input_data.weather,
            "Driving_Style": input_data.driving_style,
            "Laps_Remaining": input_data.laps_remaining,
            "Tire_Compound": predicted_tire,
            "Prediction_Time": datetime.datetime.now().isoformat()
        }
        collection.insert_one(record)

        return {"tire_compound": predicted_tire}

    except Exception as e:
        return {"error": str(e)}
    
from fastapi import HTTPException
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score




@app.get("/upload")
async def get_training_data():
    """
    Retrieve all available race data from MongoDB for model retraining
    """
    try:
        # Fetch all documents from the collection
        data_cursor = collection.find({}, {"_id": 0})
        data_list = list(data_cursor)
        
        if not data_list:
            return {
                "status": "No data available",
                "records_found": 0,
                "data": []
            }
        
        return {
            "status": "Data retrieved successfully",
            "records_found": len(data_list),
            "data": data_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch data: {str(e)}"
        )



@app.post("/retrain")
async def retrain_model():
    try:
        start_time = datetime.datetime.now()
        
        # 1. Fetch and validate data
        data_cursor = collection.find({}, {"_id": 0})
        df = pd.DataFrame(list(data_cursor))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data available for retraining")

        # 2. Preprocess data
        label_encoders = {}
        categorical_cols = ["Weather", "Driving_Style", "Tire_Compound"]
        numerical_cols = ["Lap", "Temperature", "Laps_Remaining"]
        
        # First encode Tire_Compound
        le = LabelEncoder()
        df["Tire_Compound"] = le.fit_transform(df["Tire_Compound"])
        label_encoders["Tire_Compound"] = le
        num_classes = len(le.classes_)
        
        # Then encode other categorical features
        for col in ["Weather", "Driving_Style"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Scale numerical features
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # 3. Prepare features and labels
        feature_cols = numerical_cols + ["Weather", "Driving_Style"]
        X = df[feature_cols].values
        y = df["Tire_Compound"].values

        # 4. Model architecture
        try:
            model = tf.keras.models.load_model("models/race_strategy_model.h5")
            if model.layers[-1].output_shape[-1] != num_classes:
                raise ValueError("Loaded model has incorrect output dimension")
            print("Loaded existing model for retraining")
        except Exception as e:
            print(f"Creating new model. Reason: {str(e)}")
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

        # 5. Train model with validation split
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X, y,
            epochs=100,
            batch_size=16,
            validation_split=0.2,  # Automatically splits 20% for validation
            callbacks=[early_stopping],
            verbose=0
        )

        # 6. Evaluate using the validation data from history
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        # For more detailed metrics, we'd need to manually split the data
        # Alternative evaluation approach:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # 7. Save artifacts
        os.makedirs("models", exist_ok=True)
        model.save("models/race_strategy_model.h5")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(label_encoders, "models/label_encoders.pkl")

        # 8. Store logs
        training_duration = (datetime.datetime.now() - start_time).total_seconds()
        
        training_logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "loss": history.history['loss'][-1],
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "epochs": len(history.history['loss']),
            "training_samples": len(X_train),
            "validation_samples": len(X_test),
            "training_time": training_duration,
            "classification_report": class_report,
            "model_version": f"v{len(training_logs) + 1}",
            "features_used": feature_cols,
            "num_classes": num_classes
        })
        
        return {
            "status": "Retraining successful",
            "accuracy": accuracy,
            "val_accuracy": val_accuracy,
            "training_time_seconds": training_duration,
            "new_samples_used": len(df),
            "model_version": training_logs[-1]["model_version"],
            "early_stopped": len(history.history['loss']) < 100
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


from fastapi.responses import JSONResponse

# Global variable to store training logs
training_logs = []

@app.get("/logs")
async def get_training_logs():
    try:
        if not training_logs:
            return {"message": "No training logs available. Train a model first."}
        
        # Get the latest training log (most recent retraining)
        latest_log = training_logs[-1]
        
        # Structure the metrics response
        metrics = {
            "accuracy": latest_log.get("accuracy", None),
            "loss": latest_log.get("loss", None),
            "precision": latest_log.get("precision", None),
            "recall": latest_log.get("recall", None),
            "f1_score": latest_log.get("f1_score", None),
            "training_details": {
                "epochs": latest_log.get("epochs", None),
                "training_samples": latest_log.get("training_samples", None),
                "validation_samples": latest_log.get("validation_samples", None),
                "training_time": latest_log.get("training_time", None)
            },
            "last_trained": latest_log.get("timestamp", None)
        }
        
        return JSONResponse(content=metrics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Store logs in a list
training_logs = []


async def get_logs():
    def stream_logs():
        last_index = 0
        while True:
            if last_index < len(training_logs):
                yield f"data: {training_logs[last_index]}\n\n"
                last_index += 1
            time.sleep(1)  # Prevent CPU overuse

    return StreamingResponse(stream_logs(), media_type="text/event-stream")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin if needed (e.g., "http://localhost:5500")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)