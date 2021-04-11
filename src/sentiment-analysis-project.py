'''
Sentiment analysis on airline tweets.  
Capstone 2 for the Galvanize Data Science Immersive.
'''
# import model-testing-functions
# import nlp-functions
# import eda-functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
import nltk.corpus
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import re
import collections

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold



'''
Data Wrangling and EDA
'''

def data_overview(df):
    '''
    Prints the following to get an overview of the data for starting EDA:
        First five rows (.head())
        Shape (.shape)
        All columns (.columns)
        Readout of how many non-null values and the dtype for each column (.info())
        Numerical column stats (.describe())
        Sum of unique value counts of each column
        Total null values per column
        Total duplicate rows

    Parameter
    ----------
    df:  pd.DataFrame 
        A Pandas DataFrame

    Returns
    ----------
       None
    '''
    print("\u0332".join("HEAD "))
    print(f'{df.head()} \n\n')
    print("\u0332".join("SHAPE "))
    print(f'{df.shape} \n\n')
    print("\u0332".join("COLUMNS "))
    print(f'{df.columns}\n\n')
    print("\u0332".join("INFO "))
    print(f'{df.info()}\n\n')
    print("\u0332".join("UNIQUE VALUES "))
    print(f'{df.nunique()} \n\n')
    print("\u0332".join("NUMERICAL COLUMN STATS "))
    print(f'{df.describe()}\n\n')
    print('\u0332'.join("TOTAL NULL VALUES IN EACH COLUMN "))
    print(f'{df.isnull().sum()} \n\n')
    print('\u0332'.join("TOTAL DUPLICATE ROWS "))
    print(f' {df.duplicated().sum()}')

all_data = pd.read_csv('/Users/bn/Galvanize/Twitter-Sentiment-Analysis/data/Tweets.csv')

df = all_data.drop(columns=['tweet_id','airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline_sentiment_gold', 'negativereason_gold'])

df.drop_duplicates(inplace=True)

df.columns = ['sent', 'airline', 'name', 'rts', 'tweet', 'coords', 'time', 'location', 'timezone']

df['sent'] = df['sent'].map({'positive':1, 'negative':-1, 'neutral':0})

corpus = ''
for text in df.tweet:  
    corpus += ''.join(text) + ' '

list_words_raw_corpus = corpus.split()



'''
NLP
'''

# stopwords are nltk stopwords
# add new stopwords
StopWords = set(stopwords.words('english'))
add_stopwords = {'flight', 'virgin america', 'virginamerica', 'united', 'southwest', 'southwestair', 'delta', 'us airways', 'usairways', 'american', 'americanair', 'aa', 'jet blue', 'jetblue', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'twenty four', '@virginamerica', '@united', '@southwest', '@delta', '@usairways', '@americanair', '@jetblue', '@delta', 'amp'}
StopWords = StopWords.union(add_stopwords)

# text cleaning pipeline
def lowercase_text(text):
    return text.lower()

def remove_nums_and_punctuation(text):
    punc = '!()-[]{};:\\,<>./?@#$%^&*_~;1234567890'
    for ch in text:
        if ch in punc:
            text = text.replace(ch, '')
    return text

def remove_newlines(text):
    text.replace('\n', '')
    return text

def remove_urls(text):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())

def split_text_into_words(text):
    return text.split(' ')

def remove_stopwords(word_lst):
    return [word for word in word_lst if word not in StopWords]

def lemmatize_word_list(word_lst):
    lemmatizer = WordNetLemmatizer()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in word_lst])
    return lemmatized

def word_list_to_string(word_lst):
    return ''.join(word_lst)

def text_cleaner(text, additional_stop_words=[]):
    text_lc = lowercase_text(text)
    text_np = remove_nums_and_punctuation(text_lc)
    text_nnls = remove_newlines(text_np)
    text_nurl = remove_urls(text_nnls)
    words = split_text_into_words(text_nurl)
    words_nsw = remove_stopwords(words)
    lemmatized = lemmatize_word_list(words_nsw)
    cleaned_text = word_list_to_string(lemmatized)
    return cleaned_text

cleaned_corpus = text_cleaner(corpus)


def create_word_cloud(text, 
                width=700, 
                height=700, 
                background_color='black', 
                min_font_size=12
                ):
    '''
    Generates word cloud of text.
    Parameter
    ----------
    text:  str 
        A string of words to create the word cloud from.
    width:  int 
        Width in pixels of word cloud image.
    height:  int 
        Height in pixels of word cloud image.
    background_color:  str
        Color of background of word cloud image
    min_font_size:  int
        Minimum font size of 
    Returns
    ----------
    Word cloud image.
    '''
    return WordCloud(
        width=width, 
        height=height,
        background_color=background_color,
        min_font_size=min_font_size).generate(text)




cleaned_corpus_wordcloud = create_word_cloud(cleaned_corpus)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(cleaned_corpus_wordcloud)
plt.axis("off")
plt.tight_layout(pad = 1)
# plt.show()
# plt.savefig('cleaned corpus word cloud')

def common_words_graph(text, num_words=15, title='Most Common Words'):
    '''
    Horizontal bar graph of most common words in text with their counts in descending order.

    Parameter
    ----------
    text:  str 
        Text to find most common words in.
    num_words:  int
        Number of most common words to graph.
    title:  str
        Title of graph.

    Returns
    ----------
    Horizontal bar graph with counts of the most common words. Saves PNG file of the graph graph showing specified number of most common words in the passed in text string.
    '''
    txt = text.split()
    sorted_word_counts = collections.Counter(txt)
    sorted_word_counts.most_common(num_words)
    most_common_words = sorted_word_counts.most_common(num_words)
    word_count_df = pd.DataFrame(most_common_words,columns=['Words', 'Count'])

    fig, ax = plt.subplots(figsize=(8, 8))
    word_count_df.sort_values(by='Count').plot.barh(x='Words',
                      y='Count',
                      ax=ax,
                      color='deepskyblue',
                      edgecolor='k')

    ax.set_title(title)
    plt.legend(edgecolor='inherit')
    # plt.show()
    # return plt.save('common words graph')



# Lexical diversity
# total words, total unique words, average repitition, proportion of unique words
def lexical_diversity(text):
    txt = text.split()
    print(f'Total words: {len(txt)}')
    print(f'Total unique words: {len(set(txt))}')
    print(f'Average word repetition: {round(len(text)/len(set(text)), 2)}')
    return f'Proportion of unique words: {round(len(set(txt)) / len(txt), 2)}'

# see examples of the context of a selected word
def get_context(text, word, lines=10):
    '''
    See the context of a word in your corpus.

    Parameter
    ----------
    text:  str 
        String of text.  The corpus.
    word:  str
        Word to find see .
    lines:  int
        Number of lines to return.

    Returns
    ----------
    Specified number of lines each showing the context of the specified word.
    '''
    txt = nltk.Text(text.split())
    return txt.concordance(word, lines=lines)



'''
Supervised learning classification
'''

# count vectorizer (min_df ignores terms appearing in less than n docs.  max_df ignores words that are in more than n docs. min_df and max_df can take abs numbers or proportion)
cv = CountVectorizer(stop_words='english')

# count vectorizer term frequency matrix
# count_vec_tfm = cv.fit_transform(tweets_lst).toarray()
# feature_names = cv.get_feature_names()
# feature_frequencies = np.sum(count_vec_tfm)

# count vector vocabulary
# print(cv.vocabulary_)

# number of docs and unique words of the term frequency matrix
# print(cv.shape)

# see which words have become stopwords for vectorizer
# print(cv.stop_words_)

# apply text cleaner to each tweet
df['tweet'] = df['tweet'].apply(text_cleaner)


# train/test split, stratify for unbalanced classes
X = df.tweet
y = df.sent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)

# apply count vectorizer to train/test data
def vec_it(text, vectorizer=CountVectorizer):
    vectorizer = vectorizer
    words = text.split()

    vecced_text = vectorizer.fit_transform(words)
    return vecced_text

X_train = cv.fit_transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

# apply tfidf vectorizer to train/test data
#  Tfidf vectorizer
# tv = TfidfVectorizer()
# X_train = tv.fit_transform(X_train)
# y_test = tv.fit_transform(y_test)

#list classification models to test, no tuning
# models = [MultinomialNB(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier() SVC(C=6, class_weight='balanced')]

# tuned model list
models = [MLPClassifier(hidden_layer_sizes=500, activation='relu', solver='adam', alpha=.05, batch_size=10, learning_rate='adaptive'), RandomForestClassifier(n_estimators=12000, max_features=3), SVC(C=6, class_weight='balanced')]

# score list of models, return accuracy and f1 scores for each model
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

# confusion matrix for best model
def conf_matrix(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)

# cross validation for the best model
# use stratified Kfold for unbalanced classes
# need to fix this function: KeyError: 'Passing list-likes to .loc or [] with any missing labels is no longer supported, see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike'

def stratified_k_fold(model, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds)
    scores = []
    models = []
    
    for train_index, test_index in skf.split(X_train, y_train):
        X_t = X_train[train_index]
        X_v = X_train[test_index]
        y_t = y_train[train_index]
        y_v = y_train[test_index]

        model.fit(X_t, y_t)
        models.append(model)
        scores.append(model.score(X_v, y_v))

    scores = np.array(scores)
    scores_mean = scores.mean()
    best_model = models[np.argmax(scores)]
    val_score = best_model.score(X_test, y_test)

    return f'scores mean: {scores_mean} \n val score: {val_score}'



'''
Unsupervised Learning.  This code is currently incomplete.
'''

# term frequency matrix has thousands of features.  reduce features using pca?
# k means clustering to find clusters, evaluate clusters, 


# use tfidf vectorizer on cleaned corpus to get term frequency matrix
# tv = TfidfVectorizer()
# corpus_tfm = tv.fit_transform(df.tweet)
# # term frequency matrix is shape (14503, 13523)

# # use pca to regularize term frequency matrix
# pca = TruncatedSVD(100)
# truncated = pca.fit_transform(corpus_tfm)
# # truncated tfm shape is 14503, 100


# # find clusters with K Means Clustering
# kmeans = KMeans(n_clusters=3, n_jobs=-1)
# kmeans.fit(truncated)
# kmeans.cluster_centers.shape is 3,100



# # # plot kmeans
# label = kmeans.fit_predict(truncated)

# # #filter rows
# filtered_label1 = df[label == 1]
# filtered_label2 = df[label == 2]
# filtered_label3 = df[label == 3]

# plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'blue')
# plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
# plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'black')
# plt.show()




