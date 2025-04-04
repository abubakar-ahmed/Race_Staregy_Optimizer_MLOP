# Race_Staregy_Optimizer_MLOP
---
# Youtube link: https://youtu.be/YCMgyP5ctNE

## **Project Description**

The **Race Strategy Optimizer** is a web application that predicts the **optimal tire compound** (`soft`, `medium`, `hard`, `wet` or `intermidiate`) based on a combination of race parameters. It leverages a neural network classification model trained on a synthetic dataset to simulate real-world Formula 1 decision-making.  

The application includes:

-  Real-time tire prediction based on multiple dynamic inputs.
-  A custom-trained neural network built from scratch.
-  A live FastAPI backend.
-  Docker containerization for seamless deployment.
-  Load testing via **Locust** to evaluate performance under stress.

---

## 📽️ **Demo Video**

Watch the full project demo here:  
# 🔗  Youtube link: https://youtu.be/YCMgyP5ctNE

---

# **Live Links**

### **Live API Endpoint (FastAPI)**
Link: https://race-staregy-optimizer-mlop.onrender.com/docs

### **Frontend Web App**
Link: https://race-staregy-optimizer-mlop.onrender.com/home

---

## **Setup Instructions**

### Clone the Repository
```bash
git clone https://github.com/yourusername/race-strategy-optimizer.git
cd race-strategy-optimizer
```
---
### Run Locally

> Python 3.9+ and `pip` must be installed.

#### 1. Create a virtual environment:
```bash
python -m venv venv
```

#### 2. Activate it:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. Install the requirements:
```bash
pip install -r requirements.txt
```

#### 4. Run the FastAPI app:
```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the Swagger UI!

---

## **Flood Request Simulation Results**

Load testing was conducted using **Locust** to simulate high traffic and analyze the system's robustness.

### 📊 Summary:
- [Flood Simulation Results](/Locust_logs.png)  
- **Visualizations:**  
  ![Flood Simulation](/Locust_charts.png)
  ![Flood Simulation](/Locust_logs.png)

# Explanation 
These images above showcase the results of the Flood Request Simulation using Locust to evaluate the API's performance under load.

# Locust Test Report (Locust_logs.png)
  - Provides detailed request statistics, including the total number of requests, response time, and failure rate.
  - Confirms that all requests to /predict and /retrain endpoints were successful with zero failures.
  - Highlights response time metrics, showcasing latency at different percentiles.

# Performance Charts (Locust_charts.png)
  - Total Requests per Second: Shows API request throughput over time.
  - Response Times (ms): Displays how response times evolved, with percentile-based analysis.
  - Number of Users: Visualizes the number of concurrent users simulated during testing.

These results validate the robustness and scalability of the deployed API under simulated load conditions.

### 📁 Files:
-  [Flood Simulation Results](/Locust_logs.png) 
- Graph Example:  
  ![Flood Simulation](/Locust_charts.png)

---

## 🔍 **Prediction Example**

| Feature             | Value      |
|---------------------|------------|
| Lap             | 30      |
| Tempreture     | 35      |
| Driving Style       | Aggressive |
| Weather    | Rainy        |
| Laps Remaining           | 10   |
| **Prediction**      | **Wets** |

---
## Author: Abubakar Ahmed Umar
---
