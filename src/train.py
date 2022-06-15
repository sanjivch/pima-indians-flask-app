#Importing basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#Importing the Dataset
df = pd.read_csv('https://raw.githubusercontent.com/curiousily/Logistic-Regression-with-TensorFlow-js/master/src/data/diabetes.csv')


#Replacing the zero-values for Blood Pressure
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'BloodPressure':0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure':0}, np.median(df2['BloodPressure']))
dataframe = [df1, df2]
df = pd.concat(dataframe)

#Splitting the data into dependent and independent variables
X = df.drop('Outcome', axis = 1)
Y = df['Outcome']
columns = X.columns

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)
scaler_filename = 'diabetes_scaler_rf.pkl'
pickle.dump(scaler, open(scaler_filename, 'wb'))

X = pd.DataFrame(X_transformed, columns = columns)

#Splitting the data into training and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


smt = SMOTE()
X_train, y_train = smt.fit_resample(X_train, y_train)


model = RandomForestClassifier(n_estimators=300, bootstrap = True, max_features = 'sqrt')
model.fit(X_train, y_train)


model_filename = 'diabetes_model_rf.pkl'
pickle.dump(model, open(model_filename, 'wb'))


y_pred = model.predict(X_test)
print(X_test)
print(scaler.inverse_transform(X_test)[0])

print(X_test.columns)
print(X_test.head())

#print(list(zip(y_test, y_pred)))

# print('Accuracy of Random Forest on test set: {:.2f}'.format(model.score(X_test, y_test)))
# print(f1_score(y_test, y_pred, average="macro"))
# print(precision_score(y_test, y_pred, average="macro"))
# print(recall_score(y_test, y_pred, average="macro"))