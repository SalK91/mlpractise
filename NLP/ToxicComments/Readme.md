# Toxic Comment Classification Challenge

Identify and classify toxic online comments

## Data Set:
- From Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


## Feature Generation from text:
-  Lemmitization via NLTK.
-  Count Vectorizer via Sklearn
-  Tf-idf transformation via Sklearn.

## Machine Learning Classification Models
-  ML Models and Hyper Parameter Search via TPOT Classifier (Sparse Config.)

### Model Accuracy Results:
- Model 1: Manual Data Processing and LR Models with L2 regularization.
  - Accuracy = 0.90
  - Precision = 0.33
  - Recall = 0.03
  - F-1 score = 0.061

- Model 2: Sklearn Count Vectorizer and Linear SVC Models with L1 regularization.
  - Accuracy = 0.91
  - Precision = 0.54
  - Recall = 0.79
  - F-1 score = 0.64

- Model 3: Sklearn Count Vectorizer and Naive Bayes.
  - Accuracy = 0.91
  - Precision = 0.53
  - Recall = 0.76
  - F-1 score = 0.63

- Model 4: Sklearn Tf-IDF and Linear SVC Models with L1 regularization.
  - Accuracy = 0.92
  - Precision = 0.58
  - Recall = 0.83
  - F-1 score = 0.68

- Model 5: Sklearn Tf-IDF (with uni/bi and tri-grams) and features limited to 5000

## Next Steps
-  Jaccardi Distance for Spelling errors?
-  Multinomial vs Binomial NB?
-  Pre-Trained Embedding Layers?
