import cv2
from sklearn.metrics import confusion_matrix, accuracy_score
from util import readDb
import os

dfTrain, dfTest = readDb()
threshold = 405

def varianceOfLaplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

labels = []
predicted = []
for index, row in dfTest.iterrows():
    path = str(row['Path']).replace("db1", "db2")
    label = int(row['Label'])
    if os.path.exists(path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = varianceOfLaplacian(gray)

        labels.append(label)

        if blur > threshold:
            predicted.append(1)
        else:
            predicted.append(0)


print(accuracy_score(y_true=labels, y_pred=predicted))
print(confusion_matrix(y_true=labels, y_pred=predicted))