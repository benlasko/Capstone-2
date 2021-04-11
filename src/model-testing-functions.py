from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def apply_vectorizer(text, vectorizer=CountVectorizer):

    vectorizer = vectorizer
    words = text.split()

    vecced_text = vectorizer.fit_transform(words)
    return vecced_text

X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()


def score_class_models(models, X_train, y_train, X_test, y_test):
    '''
    Tests a list of predictive models and returns accuracy and f1 scores for each model.

    Parameter
    ----------
    models:  list 
        A list of models to test.
    X_train:  arr
        X_train data.
    y_train:  arr
        y_train data.
    X_test:  arr
        X_test data.
    y_test:  arr
        y_test data.
        
    Returns
    ----------
    Accuracy and f1 scores for each model in the list of models passed in.
    '''
    acc_score_list = []
    f1_score_list = []

    for model in models:   
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_score_list.append(model.score(X_test, y_test))
        f1_score_list.append(f1_score(y_test, y_pred, average='weighted'))
        
    for model, score, f1_scored in zip(models, acc_score_list, f1_score_list):
        print(f'{model} accuracy: {round(score * 100, 2)} %')
        print(f'{model} f1: {round(f1_scored * 100, 2)} %')     


def conf_matrix(model, X_train, y_train):
    '''
    Returns a confusion matrix for the model.

    Parameter
    ----------
    model: object
        A predictive model
    X_train: arr
        X_train data
    y_train: arr
        y_train data
        
    Returns
    ----------
    A confusion matrix for the model passed in.
    '''

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)