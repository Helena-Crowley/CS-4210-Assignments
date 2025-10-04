#-------------------------------------------------------------------------
# AUTHOR: Helena Crowley
# FILENAME: knn.py
# SPECIFICATION: KNN algorithm to classify emails as spam or ham
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

errors = 0
classes = {"spam": 1, "ham": 2}
#Loop your data to allow each instance to be your test set
for i in range(len(db)):

    X = []
    Y = []
    for j in range(len(db)):
        if j == i:
            continue
        X.append([float(val) for val in db[j][0:20]])
        Y.append(classes[db[j][20]])

    test_sample = [float(val) for val in db[i][0:20]]
    true_label = classes[db[i][20]]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    #--> add your Python code here
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([test_sample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != true_label:
        errors += 1

#Print the error rate
#--> add your Python code here
error_rate = errors / len(db)
print("Error rate:", error_rate)





