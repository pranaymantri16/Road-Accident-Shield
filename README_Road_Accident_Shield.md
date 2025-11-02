
# 🛣️ Road Accident Shield  
**Accident Severity Prediction and Classification using Machine Learning**

## 📘 Overview
Road Accident Shield is a machine learning–powered system designed to **analyze, predict, and classify accident severity** based on real-world traffic and environmental data.  
It uses Python, Scikit-Learn, and Flask to train predictive models and provide an interactive web interface for real-time accident severity prediction.

---

## 🎯 Objectives
- Analyze key factors contributing to road accidents.  
- Build an ML model to predict the **severity of an accident** (Slight, Serious, or Fatal).  
- Deploy the model as a **Flask web app** for user interaction.  
- Provide insights that help **reduce accident risks** and support traffic authorities.

---

## 🧠 Machine Learning Workflow

### 1. **Data Sources**
The project uses real UK accident datasets:
- `accidentsBig.csv` – accident details (time, weather, road type, etc.)
- `vehiclesBig.csv` – vehicle data (type, maneuver, impact)
- `casualtiesBig.csv` – injury and casualty information  
These are merged into `combined_balanced_3000.csv` for training.

### 2. **Data Preprocessing**
- Dropping irrelevant or missing fields  
- Combining date and time into a single datetime column  
- Handling `-1` (unknown) values  
- Encoding categorical features using **LabelEncoder**  
- Normalizing features using **StandardScaler**  
- Balancing dataset using **SMOTE** (Synthetic Minority Oversampling Technique)

### 3. **Model Training**
Multiple models were tested:
- **Random Forest Classifier** (best performing)  
- Logistic Regression  
- Decision Tree Classifier  

The trained Random Forest model achieved high accuracy and was saved as `accident_severity_model.sav`.

### 4. **Evaluation**
Metrics used:
- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-Score  

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|------------------|
| Language | Python 3 |
| Libraries | pandas, numpy, scikit-learn, pickle, flask |
| Frontend | HTML, CSS (in `templates/` and `static/`) |
| Backend | Flask |
| Visualization | Matplotlib, Seaborn |
| Deployment | Localhost / Render / Railway |

---

## 🖥️ Flask Web Application

### Key Files
- `main.py` → Flask application file  
- `templates/index.html` → Main UI for user input  
- `static/index.css` → Styling for the interface  
- `litemodel.sav`, `scaler.pkl` → ML model & normalization scaler  

### Features
- User inputs factors like **Weather, Light, Road Type, Vehicle Type, etc.**  
- The app predicts **accident severity** instantly.  
- Option to visualize and export data reports.

### Run Instructions
```bash
# 1. Clone the repository
git clone https://github.com/pranaymantri16/Road-Accident-Shield.git
cd Road-Accident-Shield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Flask app
python main.py

# 4. Open in browser
http://127.0.0.1:5000/
```

---

## 📊 Model Files
| File | Description |
|-------|--------------|
| `accident_severity_model.sav` | Trained Random Forest model |
| `litemodel.sav` / `litemodel_new.sav` | Lightweight models for deployment |
| `scaler.pkl` | Feature scaling object |
| `combined_balanced_3000.csv` | Preprocessed dataset used for model training |

---

## 🧩 System Architecture
```
+------------------------+
| Raw Accident Datasets  |
+-----------+------------+
            |
            ▼
+------------------------+
| Data Cleaning & Merge  |
+-----------+------------+
            |
            ▼
+------------------------+
| Feature Engineering    |
+-----------+------------+
            |
            ▼
+------------------------+
| Model Training (RF)    |
+-----------+------------+
            |
            ▼
+------------------------+
| Flask Web Interface    |
+------------------------+
```

---

## 📈 Results
- Model Accuracy: **~89–92%**
- Random Forest performed best among tested models.
- Key influencing factors: Weather, Road Surface, Light Conditions, Vehicle Type.

---

## 🧑‍💻 Authors
**Pranay Mantri**  
Guided by: *Dr. Amit Pimpalkar*  
Department of Computer Engineering  
RCOEM

---

## 💬 Future Scope
- Integration with real-time traffic APIs.  
- Accident hotspot visualization using Google Maps API.  
- Advanced deep learning models for higher accuracy.  
- Mobile-friendly dashboard and alert system.

---