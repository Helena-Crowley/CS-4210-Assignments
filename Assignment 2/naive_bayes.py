#-------------------------------------------------------------------------
# AUTHOR: Helena Crowley
# FILENAME: naive_bayes.py
# SPECIFICATION: Naive Bayes Classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temp = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Weak": 1, "Strong": 2}
play_tennis = {"Yes": 1, "No": 2}

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
X = []
Y = []
for row in dbTraining:
    outlook_temp = outlook[row[1]]
    temp_temp = temp[row[2]]
    humidity_temp = humidity[row[3]]
    wind_temp = wind[row[4]]
    label_temp = play_tennis[row[5]]
    X.append([outlook_temp, temp_temp, humidity_temp, wind_temp])
    Y.append(label_temp)

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print(f"{'Day':<13}{'Outlook':<12}{'Temperature':<12}{'Humidity':<10}{'Wind':<8}{'PlayTennis':<12}{'Confidence':<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
    outlook_t = outlook[row[1]]
    temp_t = temp[row[2]]
    humidity_t = humidity[row[3]]
    wind_t = wind[row[4]]
    
    test_sample = [[outlook_t, temp_t, humidity_t, wind_t]]

    sample_prob = clf.predict_proba(test_sample)[0]

    sample_pred = clf.predict(test_sample)[0]
    pred_class = [k for k,v in play_tennis.items() if v == sample_pred][0]
    
    confidence = sample_prob[sample_pred - 1]
    if confidence >= 0.75:
        print(f"D{row[0]:<12}{row[1]:<12}{row[2]:<12}{row[3]:<10}{row[4]:<8}{pred_class:<12}{confidence:.2f}")
