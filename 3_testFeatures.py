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

featureName = 'glcm'
featureName = 'LBP'
featureName = '256_SIFT'
featureName = '512_SIFT'

featureName = 'LBP'

computeNearestNeighbors = True
computeSVM = True
computeSGD = True
computeNaiveBayes = True
computeDecisionTrees = True
computeAdaboost = True
computeGradientBoosting = True
computeRandomForest = True
computeExtremellyRandomForest = True

xTrainFilename = '../features/xTrain_{}.csv'.format(featureName)
xTestFilename = '../features/xTest_{}.csv'.format(featureName)
yTrainFilename = '../features/yTrain_{}.csv'.format(featureName)
yTestFilename = '../features/yTest_{}.csv'.format(featureName)

xTrain = pd.read_csv(xTrainFilename).as_matrix()
xTest = pd.read_csv(xTestFilename).as_matrix()

yTrain = pd.read_csv(yTrainFilename).values
yTest = pd.read_csv(yTestFilename).values

fileResults = '../results/{}.txt'.format(featureName)
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
cValues = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000]
if computeSVM:
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
        print(confusionMatrix)
        f.write("\nTrain {}".format(descriptorName))
        f.write('\nAccuracy: {}'.format(accuracy))
        f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

descriptorName = 'SVM RBF'
cValues = [0.01, 0.1, 1, 10, 100, 500, 1000, 2000, 4000, 10000, 2000000]
if computeSVM:
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
        print(confusionMatrix)
        f.write("\nTrain {}".format(descriptorName))
        f.write('\nAccuracy: {}'.format(accuracy))
        f.write('\nConfusion matrix:\n {}\n'.format(str(confusionMatrix)))

