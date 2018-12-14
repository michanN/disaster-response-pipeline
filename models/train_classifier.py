from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import sys
import pandas as pd
import numpy as np
import pickle
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''
    Loads table and returns X, Y variables as well as the category names.

    INPUT
        database_filepath (string): The path to the database.

    OUTPUT:
        X (ndarray): The message data - features
        Y (ndarray): The categories - targets
        categories (list): Labels for the categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', con=engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    categories = list(df.columns[4:])

    return X, Y, categories


def tokenize(text):
    '''
    Normalizes text by removing special characters, converts to lower, removes stop words
    and lemmatize words.

    INPUT:
        text (string): Text to normalize.

    OUTPUT 
        words (list): List of tokens.
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-z0-9]", " ", text.lower())

    words = word_tokenize(text)

    words = [lemmatizer.lemmatize(word, pos='v').strip()
             for word in words if word not in stop_words]

    return words


def build_model():
    '''
    Creates a ML pipeline using CountVectorizer, TFIDF, LinearSVD
    and runs GridSearchCV.

    OUTPUT:
        cv_pipeline (gridsearchcv): Result of GridSearchCV
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(random_state=12), n_jobs=-1))
    ])

    parameters = {
        'tfidf__smooth_idf': [True, False],
        'clf__estimator__C': [1, 2, 5]
    }

    cv_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters,
                               scoring='f1_micro', verbose=3, n_jobs=-1)

    return cv_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using the test data.

    INPUT:
        model (gridsearchcv): Model to be evaluated.
        X_test (ndarray): The test data - features.
        Y_test (ndarray): The true labels for test data.
        category_names (list): Labels for the categories.

    OUTPUT 
        Print metrics (accuracy, precision, recall and f1-score).
    '''
    y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        accuracy = accuracy_score(Y_test[:, i], y_pred[:, i])
        precision, recall, fscore, _ = score(
            Y_test[:, i], y_pred[:, i], average='micro')
        print('Category: {:<22} Acc: {:<8.2%} | Precision: {:<8.2%} | Recall: {:<8.2%} | Fscore: {:.2%}'.format(
            category, accuracy, precision, recall, fscore))


def save_model(model, model_filepath):
    '''
    Saves model as a pickle file.

    INPUT:
        model (): Model to be saved.
        model_filepath (string): Path to save location.
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
