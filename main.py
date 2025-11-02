from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import urllib.request
import urllib.parse
from sklearn.tree import DecisionTreeClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Fix for old DecisionTree models
DecisionTreeClassifier.monotonic_cst = None

# ------------------------------------------------
# Flask App Setup
# ------------------------------------------------
app = Flask(__name__)

# Load trained model & scaler
model = joblib.load('accident_severity_model.sav')
scaler = joblib.load('scaler.pkl')

# Optional: Email Alert Function (smtp)
def sendEmail(receiver_email, subject, message):
    sender_email = "pranaymantri13@gmail.com"
    sender_password = "abcd efgh ijkl mnop"  # Use App Passwords for Gmail

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        # Connect to Gmail SMTP
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("✅ Email sent successfully!")

    except Exception as e:
        print("❌ Email sending failed:", e)

# Prediction Function
def cal(ip):
    try:
        input_dict = dict(ip)
        print("Raw input:", input_dict)

        # Expected fields from the HTML form
        fields = [
            'Did_Police_Officer_Attend', 'age_of_driver', 'vehicle_type',
            'age_of_vehicle', 'engine_cc', 'day', 'weather',
            'roadsc', 'light', 'gender', 'speedl'
        ]

        # Convert to float and reshape for model input
        data = [float(input_dict[f][0]) for f in fields]
        print("Prepared data:", data)

        data = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data)

        # --- Extract individual fields for rule-based logic ---
        vehicle_type = int(input_dict['vehicle_type'])
        weather = int(input_dict['weather'])
        roadsc = int(input_dict['roadsc'])
        speed = float(input_dict['speedl'])
        light = int(input_dict['light'])
        age_driver = np.exp(float(input_dict['age_of_driver']))

        severity_map = {
            1: "Fatal",
            2: "Serious",
            3: "Slight"
        }

        # --- Hardcoded Rule Overrides ---
        if vehicle_type == 20 and weather == 5 and roadsc == 5:
            severity = "Fatal"

        elif speed > 100:
            severity = "Fatal"

        elif light != 1 and roadsc in [4, 5, 7]:
            severity = "Serious"

        elif age_driver < 25 and weather in [2, 5]:
            severity = "Serious"

        else:
            result = model.predict(data_scaled)
            print("Prediction:", result)
            severity = severity_map.get(int(result[0]), "Unknown")

        print("Predicted Severity:", severity)
        return f"Predicted Accident Severity: {severity}"

    except Exception as e:
        print("Error during prediction:", e)
        return "Error: " + str(e)



# ------------------------------------------------
# Flask Routes
# ------------------------------------------------
@app.route('/')
@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/visual/', methods=['GET'])
def visual():
    return render_template('visual.html')

@app.route('/export1.html')
def export():
    return render_template('export1.html')

@app.route('/sms/', methods=['POST'])
def alert():
    res = cal(request.form)
    try:
        # Send Email only for Fatal or Serious predictions
        if "Fatal" in res or "Serious" in res or "Slight" in res:
            sendEmail(
                receiver_email="pranaymantri16@gmail.com",
                subject="⚠️ Accident Alert",
                message=f"Accident Alert: {res}"
            )
    except Exception as e:
        print("Email Error:", e)

    return res

@app.route('/', methods=['POST'])
def get():
    return cal(request.form)

# ------------------------------------------------
# Run Flask App
# ------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
