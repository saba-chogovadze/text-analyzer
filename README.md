# Spam Classifier

A beginner machine learning project that classifies short messages as spam or
not spam, using TF-IDF features and a Naive Bayes model (scikit-learn).

## Setup
```bash
pip install -r requirements.txt
python spam_classifier.py
pytest
```

## How it works
1. Messages are converted into TF-IDF vectors
2. A Multinomial Naive Bayes model is trained on 75% of the data
3. Accuracy is measured on the remaining 25%

## Note
The included dataset is tiny and hand-written, so accuracy numbers are only a
demo. Try the SMS Spam Collection dataset from Kaggle or the UCI repository
for a realistic version.

## Ideas for improvement
- Use a real dataset and compare Naive Bayes with Logistic Regression
- Show a confusion matrix and precision/recall
- Wrap the model in a small web app with Flask or Streamlit
