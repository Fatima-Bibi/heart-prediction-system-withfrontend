from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


heart_data = pd.read_csv('heart_disease_data.csv')  
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)


app = Flask(__name__, template_folder='templates')
run_with_ngrok(app)


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        cp = float(request.form.get("cp"))
        trestbps = float(request.form.get("trestbps"))
        chol = float(request.form.get("chol"))
        fbs = float(request.form.get("fbs"))
        restecg = float(request.form.get("restecg"))
        thalach = float(request.form.get("thalach"))
        exang = float(request.form.get("exang"))
        oldpeak = float(request.form.get("oldpeak"))
        slope = float(request.form.get("slope"))
        ca = float(request.form.get("ca"))
        thal = float(request.form.get("thal"))

       
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

        
        input_data_reshaped = input_data.reshape(1, -1)
        prediction1 = model.predict(input_data_reshaped)

       
        result = "The Person does not have a Heart Disease" if prediction1[0] == 0 else "The Person has Heart Disease"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run()

@app.errorhandler(500)
def internal_server_error(e):
    return "Internal Server Error", 500

