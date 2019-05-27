from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

featureName = 'LBP'
C = 100
probabilityThresh = 0.62

xTrainFilename = '../features/xTrain_{}.csv'.format(featureName)
xTestFilename = '../features/xTest_{}.csv'.format(featureName)
yTrainFilename = '../features/yTrain_{}.csv'.format(featureName)
yTestFilename = '../features/yTest_{}.csv'.format(featureName)

xTrain = pd.read_csv(xTrainFilename).as_matrix()
xTest = pd.read_csv(xTestFilename).as_matrix()

yTrain = pd.read_csv(yTrainFilename).values
yTest = pd.read_csv(yTestFilename).values

clfSVM = svm.SVC(C=C, class_weight=None, gamma='auto', kernel='rbf', verbose=False, probability=True)
clfSVM.fit(xTrain, yTrain)

# Compute the accuracy of the model
valuesPredicted = clfSVM.predict_proba(xTest)

i = 0
classPredictions = []
for valuePredicted in valuesPredicted:
    print("{} {}".format(valuePredicted, yTest[i]))
    i += 1
    classPredictions.append(1 if valuePredicted[0] < probabilityThresh else 0)

accuracy = accuracy_score(y_true=yTest, y_pred=classPredictions)
confusionMatrix = confusion_matrix(y_true=yTest, y_pred=classPredictions)
print(accuracy)
print(confusionMatrix)

pickle.dump(clfSVM, open("../models/SVM_LBP.model", 'wb'))
