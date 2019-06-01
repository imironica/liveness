import os
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

import pandas as pd


#featureName = 'HOG'
#featureName = '256_SIFT'
#featureName = '512_SIFT'

featureName = 'glcm'
#featureName = 'LBP'



computeNearestNeighbors = True
computeSVM = True
computeSGD = True
computeNaiveBayes = True
computeDecisionTrees = True
computeAdaboost = True
computeGradientBoosting = True
computeRandomForest = True
computeExtremellyRandomForest = True

xTrainFilename = 'features/xTrain_{}.csv'.format(featureName)
xTestFilename = 'features/xTest_{}.csv'.format(featureName)
yTrainFilename = 'features/yTrain_{}.csv'.format(featureName)
yTestFilename = 'features/yTest_{}.csv'.format(featureName)

xTrain = pd.read_csv(xTrainFilename).values
xTest = pd.read_csv(xTestFilename).values

yTrain = pd.read_csv(yTrainFilename).values
yTest = pd.read_csv(yTestFilename).values

yTrain = yTrain.reshape((yTrain.shape[0],))
yTest = yTest.reshape((yTest.shape[0],))

if not os.path.exists('results'):
    os.mkdir('results')

fileResults = 'results/{}.txt'.format(featureName)
f = open(fileResults, "w")
# =================================================================================================#
# Nearest neighbor
# Train the model
if computeNearestNeighbors:
    noNeighbors = 3
    descriptorName = 'Nearest neighbors ({})'.format(noNeighbors)
    print("Train {}".format(descriptorName))

    clfNB = KNeighborsClassifier(n_neighbors=noNeighbors)
    clfNB.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfNB.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)

    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# =================================================================================================#
# SGD
# Train the model
if computeSGD:
    descriptorName = 'SGD'
    print("Train {}".format(descriptorName))
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# Naive Bayes
if computeNaiveBayes:
    descriptorName = 'Naive Bayes'
    print("Train {}".format(descriptorName))
    clf = GaussianNB()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# Decision trees
if computeDecisionTrees:
    descriptorName = 'Decision Tree Classifier '
    print("Train {}".format(descriptorName))
    clf = tree.DecisionTreeClassifier()
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# AdaBoost model
if computeAdaboost:
    descriptorName = 'Adaboost Classifier '
    print("Train {}".format(descriptorName))
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# Gradient Boosting Classifier
if computeGradientBoosting:
    descriptorName = 'Gradient Boosting Classifier'
    print("Train {}".format(descriptorName))
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clf.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# =================================================================================================#

# Random Forest Classifier
if computeRandomForest:
    descriptorName = 'Random Forest Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = RandomForestClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# Extremelly RandomForest Classifier
if computeExtremellyRandomForest:
    descriptorName = 'Extremelly Trees Classifier'
    print("Train {}".format(descriptorName))
    # Train the model
    clfRF = ExtraTreesClassifier(n_estimators=200, criterion="gini")
    clfRF.fit(xTrain, yTrain)

    # Compute the accuracy of the model
    valuePredicted = clfRF.predict(xTest)
    accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
    confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
    print('{}: {}'.format(descriptorName, accuracy))
    print(confusionMatrix)
    f.write("\nTrain {}".format(descriptorName))
    f.write('\nAccuracy: {}'.format(accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

# Support vector machines
descriptorName = 'SVM Linear'
# keep only max value 
cValues = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000]
if computeSVM:
    max_accuracy = 0
    for cValue in cValues:
        descriptorName = 'Linear SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, kernel='linear', verbose=False, probability=True)
        clfSVM.fit(xTrain, yTrain)
        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))
        if(accuracy > max_accuracy):
            max_accuracy = accuracy
            conf_matrix = confusionMatrix
            desc_name = descriptorName
    f.write("\nTrain {}".format(desc_name))
    f.write('\nAccuracy: {}'.format(max_accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(conf_matrix)))

descriptorName = 'SVM RBF'
cValues = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 4000, 10000, 2000000]
if computeSVM:
    max_accuracy = 0
    for cValue in cValues:
        descriptorName = 'SVM with C={} '.format(cValue)
        print("Train {}".format(descriptorName))
        clfSVM = svm.SVC(C=cValue, class_weight=None,
                         gamma='auto', kernel='rbf',
                         verbose=False, probability=True)
        clfSVM.fit(xTrain, yTrain)

        # Compute the accuracy of the model
        valuePredicted = clfSVM.predict(xTest)
        accuracy = accuracy_score(y_true=yTest, y_pred=valuePredicted)
        
        confusionMatrix = confusion_matrix(y_true=yTest, y_pred=valuePredicted)
        print('{}: {}'.format(descriptorName, accuracy))
        if(accuracy > max_accuracy):
            max_accuracy = accuracy
            conf_matrix = confusionMatrix
            desc_name = descriptorName
    f.write("\nTrain {}".format(desc_name))
    f.write('\nAccuracy: {}'.format(max_accuracy))
    f.write('\nConfusion matrix:\n {}\n'.format(str(conf_matrix)))

