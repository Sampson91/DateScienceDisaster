import sys
import re
import nltk
import pandas  as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from sklearn.externals import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='categories_table', con=engine)
    X = df.message.values
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    category_names = Y.columns.tolist()
    return (X, Y, category_names)


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return (tokens)


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__norm': ['l1', 'l2'],
        'clf__estimator__criterion': ['entropy', 'gini']
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return (pipeline)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    test = pd.DataFrame(Y_test, columns=category_names)
    prediction = pd.DataFrame(y_pred, columns=category_names)
    print("classification_report")
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print("Accuracy Score")
    for category in category_names:
        accuracy = accuracy_score(test[category], prediction[category])
        print("Accuracy score for {}: {:.3f} ".format(category, accuracy))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print('Building model...')
        model = build_model()

        print('Training model...')
        # model = joblib.load("./modelwww.pkl")
        model.fit(X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()