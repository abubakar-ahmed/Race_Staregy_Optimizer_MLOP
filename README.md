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

# 🌍 **Live Links**

### 🔧 **Live API Endpoint (FastAPI)**
Link: https://race-staregy-optimizer-mlop.onrender.com/docs

### 🖥️ **Frontend Web App**
Link: https://race-staregy-optimizer-mlop.onrender.com/home

---

## ⚙️ **Setup Instructions**

### 🔁 Clone the Repository
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

## 🧪 **Flood Request Simulation Results**

Load testing was conducted using **Locust** to simulate high traffic and analyze the system's robustness.

### 📊 Summary:
- [Flood Simulation Results](/locust_logs.png)  
- **Visualizations:**  
  ![Flood Simulation](/Locust_charts.png)


### 📁 Files:
-  [Flood Simulation Results](/locust_logs.png) 
- Graph Example:  
  ![Flood Simulation](/Locust_charts.png)g)

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
## Contributing
Pull requests and ideas are welcome. Fork the repo and submit your improvements!
---
