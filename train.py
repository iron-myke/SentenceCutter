import argparse
import joblib
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from dataset import load_dataset

with open("config.json", 'r') as config_file:
    config = json.load(config_file)

# Model template TfidVectorizer - Binary Classifier
def model(binary_classifier):
    return Pipeline([
        ("tfid_vectorizer", TfidfVectorizer(max_df=config['tfid_max_df'], 
            max_features=config['tfid_max_features'], ngram_range=(1, config['ngram_size']), 
            strip_accents='unicode', analyzer='char')),
        ("binary_classifier", binary_classifier)
    ])

# Load trained models (used in cut.py)
def load_model(model_type):
    return joblib.load(config['models'][model_type])

if __name__ == "__main__":
    dataset = load_dataset()
    train, test = train_test_split(dataset, test_size=0.3)
    
    # different binary classifiers
    models = [ 
        ('svm', model(LinearSVC())),
        ('rft', model(RandomForestClassifier())), 
        ('nb', model(MultinomialNB())),
        ('dtc', model(DecisionTreeClassifier())), 
        ('sgd', model(SGDClassifier()))
    ]

    for name, model in models:
        # training
        print("Training model {}...".format(name))
        model.fit(train['X'], train['Y'])
        print("Training Done.")

        # tests and displays metrics 
        Y_pred = model.predict(test['X'])
        print(classification_report(test['Y'], Y_pred))
        # saves model
        print('Saving model {} at {}...'.format(name, config['models'][name]))
        joblib.dump(model, config['models'][name])
        print("Done.\n")
