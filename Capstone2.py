# Capstone 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from EDA_plus import data_overview
from EDA_plus import nan_replacer

from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model as lm


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import spacy


'''
EDA

NLP

Classifiers using prelabeled data.  Because tweets are short, countvectorizer is prob better.

Unsupervised analysis

'''




all_data = pd.read_csv('data/Tweets.csv')
# all_data = pd.read_csv('data/Tweets.csv', encoding='ISO-8859-1')



# drop unwanted columns 
df = all_data.drop(columns=['tweet_id','airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline_sentiment_gold', 'negativereason_gold'])


# see general overview of data
# print(data_overview(df))


# rename columns
df.columns = ['sent', 'airline', 'name', 'rts', 'tweets', 'coords', 'time', 'location', 'timezone']
# # rename these after finished for readability/usability
# print(df.columns)


# make sentiments numerical
df['sent'] = df['sent'].map({'positive':1, 'negative':-1, 'neutral':0})



# remove duplicate rows (there's over 100 duplicate rows)
df.drop_duplicates(inplace=True)


# shorten tweet times to minutes, make datetime object
df['time'] = df['time'].str.slice(0,16)
df['time'] =  pd.to_datetime(df['time'])


# # replace nans/nulls
# 13641 nulls in coords.  replace with mode. there's only 832 unique coords
# 4733 nulls in location col.  replace with mode.
# 4820 nulls in timezone.  replace with mode.

# replace nans in location column with the mode
# for col in df['location', 'coords', 'timezone']:
#     df[col] = df[col].fillna(df[col].mode()[0])


df['location'] = df['location'].fillna(df['location'].mode()[0])
df['coords'] = df['coords'].fillna(df['coords'].mode()[0])
df['timezone'] = df['timezone'].fillna(df['timezone'].mode()[0])
# print(df.isna().sum())



# list of all tweets
all_tweets_lst = [tweet for tweet in df['tweets']]

# single string of all tweet text
corpus = ''
for text in df.tweets:  
    corpus += ''.join(text)





# create word cloud from string
def word_clouder(text, width=800, height=800, background_color='black', min_font_size=12):
    return WordCloud(width=800, height=800,
                     background_color='black',
                     min_font_size=12).generate(text)

# all_words_wordcloud = word_clouder(corpus)
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(all_words_wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 1)
# plt.show()
# plt.savefig('all_words_cloud')



























'''
clean text
word frequency
tf-idf, or not.
'''



# adding new stopwords
stopWords = set(stopwords.words('english'))

add_stopwords = {'flight'}
stopWords = stopWords.union(add_stopwords)






# text cleaning pipeline:


def lower(text):
    return text.lower()

def remove_punctuation(text):
    punc = '!()-[]{};:\,<>./?@#$%^&*_~;'

    for ch in text:
        if ch in punc:
            text = text.replace(ch, '')
    return text

def remove_newline(text):
    text.replace('\n', '')
    return text

def split_text_into_words(text):
    return text.split(' ')

# stop_words = set(stopwords.words("english"))

def remove_stopwords(word_lst):
    return [word for word in word_lst if word not in stopWords]

def remove_additional_stopwords(word_lst, additional_stop_words):
    return [word for word in word_lst if word not in additional_stop_words]

def lemmatizer(word_lst):
    lemmatizer = WordNetLemmatizer()
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in word_lst])
    return lemmatized

def create_cleaned_textline_from_words(words):
    return ' '.join(words)

def string_from_list(word_lst):
    return ''.join(word_lst)


def text_cleaner(text, additional_stop_words=[]):
    text_lc = lower(text)
    text_np = remove_punctuation(text_lc)
    text_nnls = remove_newline(text_np)
    words = split_text_into_words(text_nnls)
    words_nsw = remove_stopwords(words)
    # words_nasw = remove_additional_stopwords(words_nsw, additional_stop_words)
    lemmatized = lemmatizer(words_nsw)
    cleaned_text = string_from_list(lemmatized)
    return cleaned_text




tester = text_cleaner(corpus[:2000])


# after cleaning text, use countvectorizer to create term frequency matrix

# initialize vectorizer, min_df=2 ignores terms appearing in less than 2 docs, use max_df to choose to ignore words that are too frequent.  min and max df can take absolute numbers (4,5) or proportion (.5)

cv = CountVectorizer(stop_words='english', min_df=2, max_df=1000)
tfm = cv.fit_transform(all_tweets_lst)


feature_names = cv.get_feature_names()
feature_frequencies = np.sum(tfm.toarray(), axis=0)


# many numbers are flight numbers.  need to remove flight numbers but not minute descriptions( 45 mins). 


# see which words are part of the vocab
# print(cv.vocabulary_)

# see numbr of docs and unique words of the tfm
# print(cv.shape)

# see which words have become stopwords after running cv
# print(cv.stop_words_)









'''
Supervised learning Classification models

Try naive bayes, svc, randomforest, neural network:  mlpregressor, logistic regression, LSTM, 
'''

# train/test split
X_train, y_train, X_test, y_test = df.tweets[:10150], df.sent[:10150], df.tweets[10150:], df.sent[10150:]

#test models, accuracy and f1 score
model_types = [GaussianNB(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 275, max_features=5), lm.LogisticRegression(), lm.LinearRegression()]


def test_class_models(models=model_types):
    score_list = []
    for i in model_types:
        
        model = i
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score_list.append(model.score(X_test, y_test))
        
    for model, score in zip(model_types, score_list):
        print (f'{model}: {score}')

    return



# precision_score = []
# recall_score = []
# f1_score = []

# precision_score.append(precision_score(y_test, y_predict))
# recall_score.append(recall_score(y_test, y_predict))
# f1_scores = f1_score(y_true, y_pred, average=None)
