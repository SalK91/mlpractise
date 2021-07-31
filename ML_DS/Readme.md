-- Colab Notebook

# Online Shopping behavior

## Data Source:
Online Shoppers Purchasing Intention Dataset
Data Set https://www.kaggle.com/henrysue/online-shoppers-intention


## 1: Supervised Model
Build a predictive classification model (ensuring optimal features and classifier).

a. Fit a best Linear and non-linear classifier.
b. Hyper-parameter fitting process.
c. Comparison with T-POT model search

### Results
The training data after feature selection has m ~30 features and number of training records n ~ 7K. For this scale of data and features generarlly logistic regression and SVM are advised.

For the problem we would like to accurate in terms of prediction of right classes an would like a model which is reasonable in both precision and recall and hence f1-score is the choice of metric for the most optimal model.

So for our model search three set of Models are selected for this problems. For each we are ran 5 cross-validations sets across set of values of hyper-parameters to find an optimal model.

#### Model - 1 Logistic Regression:
Params = {{'C': 4.281332398719396, 'penalty': 'l1', 'solver': 'liblinear'}
Best F1 Score = 0.448
#### Model - 2 Random Forest Classifier
Params = {'class_weight': None, 'criterion': 'entropy', 'max_features': 10, 'n_estimators': 200}
Best F1 Score = 0.567
#### Model - 3 Linear SVM [BEST LINEAR MODEL ON CVs]
Params = {'C': 2, 'class_weight': 'balanced', 'penalty': 'l2'}
Best F1 Score = 0.585
Results of the best linear model on the test-set

Based on CV results Model 3 - Linear SVM was found as the best model for this problem. Result of that models on the test set are as follows:
Accuracy = 0.930
Precision = 0.597
Recall = 0.759
F1-score = 0.668
Confusion Matrix is: [[1796 100] [ 47 148]]

#### Model - 4 Non-Linear SVM [BEST NON-LINEAR MODEL ON CVs]
{'C': 1000, 'class_weight': 'balanced', 'gamma': 0.0001, 'kernel': 'rbf'}
Best F1 Score = 0.617
Results of the best non-linear SVM on the test-set
Based on CV results Model 4 - Linear SVM was found as the best model for this problem. Result of that models on the test set are as follows:

Accuracy = 0.945
Precision = 0.668
Recall = 0.826
F1-score = 0.739
Confusion Matrix is: [[1816 80] [34 161]]


#### Comparison with T-POT Grid Search
TPOT - Pipeline:

MinMaxScaler(),
RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.6, min_samples_leaf=8, min_samples_split=12, n_estimators=100)
Results on Test-set:

Accuracy = 0.945,
Precision = 0.713,
Recall = 0.677,
F1-score = 0.695
Confusion Matrix: [[1843 53] [ 63 132]]

## 2: Unsuperivsed model
User-bahavior clusters based on the purchasing behavior data for the complete dataset.

# 3: Semi-supervised Learning
Improve performance with un-labeled data in training?
