#-------------------------------------------------------------------------
# AUTHOR: Helena Crowley
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

#define features and classes
features = {
"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3,
"Myope": 1, "Hypermetrope": 2,
"No": 1, "Yes": 2,
"Reduced": 1, "Normal": 2
}
lenses = {"Yes": 1, "No": 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #read the training data in
    df_train = pd.read_csv(ds)
    for _, row in df_train.iterrows():
        dbTraining.append(row.tolist())

    #translate to numbers and add to X
    for row in dbTraining:
        X.append([
            features[row[0]],
            features[row[1]],
            features[row[2]],
            features[row[3]] 
        ])
        Y.append(lenses[row[4]])

    accuracies = []

    #Loop your training and test tasks 10 times here
    for i in range (10):

        #create tree and fit to data
       clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
       clf = clf.fit(X, Y)

       correct_predictions = 0
       for data in dbTest:
            #create test instance
            test_features = [
                features[data[0]],
                features[data[1]],
                features[data[2]],
                features[data[3]]
            ]

            #make the prediction
            class_predicted = clf.predict([test_features])[0]
            #get the true label
            true_label = lenses[data[4]]

            if class_predicted == true_label:
                correct_predictions += 1

        #get accuracy and add to list
    accuracy = correct_predictions / len(dbTest)
    accuracies.append(accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f'final accuracy when training on {ds}: {average_accuracy}')



