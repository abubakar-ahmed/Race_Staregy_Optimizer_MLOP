from locust import HttpUser, task, between
import random

class RaceOptimizerUser(HttpUser):
    wait_time = between(1, 3)  # Wait between requests

    # Sample test data
    test_data = {
        "lap": random.randint(1, 60),
        "temperature": random.randint(20, 40),
        "weather": random.choice(["Sunny", "Rainy", "Cloudy", "Mixed"]),
        "driving_style": random.choice(["Aggressive", "Balanced", "Conservative"]),
        "laps_remaining": random.randint(1, 60)
    }

    @task(1)
    def predict_tire(self):
        """Simulate a user sending a prediction request."""
        self.client.post("/predict", json=self.test_data)

    @task(2)
    def retrain_model(self):
        """Simulate a retraining request with a lower frequency."""
        self.client.post("/retrain")