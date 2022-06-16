from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle



app = Flask(__name__)

model = pickle.load(open('diabetes_model_rf.pkl', 'rb'))
scaler = pickle.load(open('diabetes_scaler_rf.pkl', 'rb'))


@app.route('/')
def home():
    # Feature list split into two for the rendering (aesthetic) reasons 
    feature_list_1 = ['Age', 'BMI', 'Glucose', 'Insulin']
    feature_list_2 = ['BloodPressure', 'SkinThickness', 'Pregnancies', 'DiabetesPedigreeFunction']
    feature_ranges = {'Pregnancies' : "0-10", 
                      'Glucose': "60-500 mg/dL", 
                      'BloodPressure': "40-200 mmHg", 
                      'SkinThickness' : "0-50 mm", 
                      'Insulin' : "0-200 mu U/ml", 
                      'BMI': "16-40", 
                      'DiabetesPedigreeFunction': "0-1", 
                      'Age':"1-99 years"}
    predict_flag = 0
    return render_template('index.html', feature_list_1=feature_list_1, feature_list_2=feature_list_2, feature_ranges=feature_ranges, predict_flag=predict_flag)

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
    predict_flag = 1
    
    return render_template('result.html', feature_list=feature_list, predict_flag=predict_flag, pred=pred)

if __name__ == "__main__":
    app.run(debug=True)