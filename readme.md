### 1. Short description

This repository contains a set of experiments for liveness detection.


### 2. Steps:

1) run 1_extractFaces.py  - extracts the faces from test images


2) Feature extraction:

   - run *2_computeHaralickFeatures.py'*  extracts the Haralick features. See more details on http://iab-rubric.org/papers/BTAS16-Anti-Spoofing.pdf
   
   - run *2_computeBoWFeatures.py --dictionarySize 512 --descriptorType SIFT*  extract Bag of Words features (expects two parameters: descriptor type and dictionary size)
   
   - run *2_computeHofFeatures.py*  extracts the HoG features. 
   
   - run *2_computeLBPFeatures.py'*  extracts the LBP features. 
   
3) Test features:
-  run *3_testFeatures.py*  compute the accuracy performance for each feature / classifier (Nearest Neighbors, SVM, SGD, Naive Bayes, Decision Trees, Adaboost, Gradient Boosting, Random Forest, Extremelly RandomForest)

4) Create a net from scratch
   
5) TODO: Transfer learning with MobileNet ang Resnet50

### 3. Results

TODO 
   
   