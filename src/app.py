from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('diabetes_model_rf.pkl', 'rb'))
scaler = pickle.load(open('diabetes_scaler_rf.pkl', 'rb'))


@app.route('/')
def home():
    feature_list = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    return render_template('index.html', feature_list=feature_list)

@app.route('/predict', methods=['POST'])
def predict_diabetes():

    # Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    #    'BMI', 'DiabetesPedigreeFunction', 'Age'],
    feature_list = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    data_list = []
    for feature in feature_list:
        data_list.append(float(request.form[feature]))
    
    X_input = np.array([data_list])

    print(X_input)
    X_input = scaler.transform(X_input)
    print(X_input)
    pred = model.predict(X_input)
    print(pred)
    return render_template('result.html', pred=pred)

if __name__ == "__main__":
    app.run(debug=True)