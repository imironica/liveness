from util import testFaceRecognitionAlgorithm, saveFaces, testDlibFaceDetector, saveFaces_hogFaceDetector

# choose method to detect faces: HAAR Cascade or HOG algorithm
if __name__ == '__main__':
    # testFaceRecognitionAlgorithm() #HAAR Cascade
    # testDlibFaceDetector() # HOG
    # saveFaces()
    saveFaces_hogFaceDetector()
