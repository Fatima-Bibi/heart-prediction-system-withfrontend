from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('other.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()

model.fit(X_train, Y_train)
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

input_data_as_numpy_array= np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        age = float(request.form.get("age"))
        
        input_data = np.array([age, ...])  
        input_data_reshaped = input_data.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)

        
        result = "The Person does not have a Heart Disease" if prediction[0] == 0 else "The Person has Heart Disease"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("home.html", result=result)


if __name__ == "__main__":
    app.run()
