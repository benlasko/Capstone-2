# Capstone 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import re

from sklearn.preprocessing import StandardScaler

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


'''
EDA

NLP

Classifiers using prelabeled data

Unsupervised analysis

'''

def data_overview(df):
    '''
    Overview of dataframe to start EDA.

    Parameter
    ----------
    df:  pd.DataFrame 
        A Pandas DataFrame

    Returns
    ----------
        First five rows (.head())
        Shape (.shape)
        All columns (.columns)
        Readout of how many non-null values and the dtype for each column (.info())
        Numerical column stats (.describe())
        Sum of unique value counts of each column
        Total of NaN/null values per column
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
    print('\u0332'.join("TOTAL NULL VALUES IN EACH COLUMN"))
    print(f'{df.isnull().sum()} \n\n')
    return



all_data = pd.read_csv('data/Tweets.csv')
# all_data = pd.read_csv('data/Tweets.csv', encoding='ISO-8859-1')


# drop unwanted columns 
df = all_data.drop(columns=['tweet_id','airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline_sentiment_gold', 'negativereason_gold'])


# see general overview of data
# print(data_overview(df))


# remove duplicate rows (there's over 100 duplicate rows)
df.drop_duplicates(inplace=True)


# rename columns
df.columns = ['sent', 'airline', 'name', 'rts', 'tweet', 'coords', 'time', 'location', 'timezone']
# print(df.columns)


# shorten times to the minute and remove timezones, make datetime object
df['time'] = df['time'].str.slice(0,16)
df['time'] = pd.to_datetime(df['time'])


# fill nans/nulls
# 13641 nulls in coords.  replace with mode although there's only 832 unique coords
# 4733 nulls in location col.  replace with mode.
# 4820 nulls in timezone.  replace with mode.

# replace nans in location column with the mode
# for col in df['location', 'coords', 'timezone']:
#     df[col] = df[col].fillna(df[col].mode()[0])

df['location'] = df['location'].fillna(df['location'].mode()[0])
df['coords'] = df['coords'].fillna(df['coords'].mode()[0])
df['timezone'] = df['timezone'].fillna(df['timezone'].mode()[0])
# print(df.isna().sum())



# EDA plots

# total sentiment counts plot
# ax = sns.catplot(x="sent", kind="count", palette="colorblind", data=df, order=['negative', 'neutral', 'positive'])
# ax.set(xlabel="Sentiment", ylabel = "Count", title='Total Sentiment Counts')
# plt.show()
# plt.savefig('Sentiment Counts')


# sentiment counts by airline
ax = df.groupby(['airline','sent'])['sent'].count().unstack(0).plot.bar(figsize=(10,10), edgecolor='k')
ax.set_title('Sentiment Counts for each Airline', size=20)
ax.set_xlabel('Sentiment')
ax.set_ylabel('Counts')
plt.show()




# other plots:
# neg/pos sentiment by day, neg/pos sentiment by day by airline
# wordcloud of raw tweets (done below)
# neg/pos sent by timezone
# top 20 words and their counts



# make sentiments numerical (after EDA plots)
df['sent'] = df['sent'].map({'positive':1, 'negative':-1, 'neutral':0})







# list of all tweets
tweets_lst = [tweet for tweet in df['tweet']]

# single string of all tweet text
corpus = ''
for text in df.tweet:  
    corpus += ''.join(text)





# word cloud function
def word_clouder(text, width=800, height=800, background_color='black', min_font_size=12):
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
    return WordCloud(width=800, height=800,
                     background_color='black',
                     min_font_size=12).generate(text)


# create word cloud of raw corpus
# corpus_wordcloud = word_clouder(corpus)
# plt.figure(figsize = (8, 8), facecolor = None)
# plt.imshow(all_words_wordcloud)
# plt.axis("off")
# plt.tight_layout(pad = 1)
# plt.show()
# plt.savefig('all_words_cloud')





# word count dictionary, don't need it
# def word_count_dict(text):
#     d = {}
#     for string in example_lst:
#         if string in d.keys():
#             d[string] += 1
#     else:
#         d[string] = 1
#     return d








# stopwords = the nltk stopwords
# add new stopwords:
StopWords = set(stopwords.words('english'))

add_stopwords = {'flight'}
StopWords = StopWords.union(add_stopwords)






# for each tweet/for entire corpus:

def lower(text):
    return text.lower()

def remove_punctuation(text):
    punc = '!()-[]{};:\\,<>./?@#$%^&*_~;'

    for ch in text:
        if ch in punc:
            text = text.replace(ch, '')
    return text

def remove_newline(text):
    text.replace('\n', '')
    return text

def remove_url(text):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())

def split_text_into_words(text):
    return text.split(' ')

# stop_words = set(stopwords.words("english"))

def remove_stopwords(word_lst):
    return [word for word in word_lst if word not in StopWords]

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
    text_nurl = remove_url(text_nnls)
    words = split_text_into_words(text_nurl)
    words_nsw = remove_stopwords(words)
    lemmatized = lemmatizer(words_nsw)
    cleaned_text = string_from_list(lemmatized)
    return cleaned_text


tester = text_cleaner(corpus[:2000])


# many numbers are flight numbers.  need to remove flight numbers but not minute descriptions( 45 mins).  remove all numbers with len>2 ?


# Part Of Speech tagging
def get_wordnet_pos(word):
    '''Map POS tag to first character lemmatize() accepts'''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# n-grams








# term frequency matrix

# count vectorizer.  min_df=2 ignores terms appearing in less than 2 docs, use max_df to choose to ignore words that are too frequent. min and max df can take abs numbers or proportion.
cv = CountVectorizer(stop_words='english', min_df=0, max_df=1000)
count_vec_tfm = cv.fit_transform(tweets_lst).toarray()
feature_names = cv.get_feature_names()
feature_frequencies = np.sum(count_vec_tfm)


#  Tfidf vectorizer
tv = TfidfVectorizer()
tfidf_tfm = tv.fit_transform(tweets_lst)









# run text cleaner on each tweet
# run vectorizer on each cleaned tweet

def vec_it(text, vectorizer=cv):
    vectorizer = vectorizer
    words = text.split()

    vecced_text = vectorizer.fit_transform(words)
    return vecced_text



# df['tweet'] = df['tweet'].apply(text_cleaner)
# print(df['tweet'][400])

# df['tweet'] = df['tweet'].apply(vec_it)
# print(df['tweet'][400])

# see which words are part of the vocab
# print(cv.vocabulary_)

# see numbr of docs and unique words of the tfm
# print(cv.shape)

# see which words have become stopwords for cv
# print(cv.stop_words_)






'''
Supervised Classification

Try naive bayes, svc, randomforest, logistic reg, neural network:  mlpregressor, logistic regression, LSTM, 
'''

# apply text cleaner
df['tweet'] = df['tweet'].apply(text_cleaner)


# train/test split
X = df.tweet
y = df.sent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)


# apply count vec to train/test data
X_train = cv.fit_transform(X_train).toarray()
X_test = cv.fit_transform(X_test).toarray()
# y_test = cv.fit_transform(y_test)


# apply tfidf vec to train/test data
# X_train = tv.fit_transform(X_train)
# y_test = tv.fit_transform(y_test)


#test models
# , SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 275, max_features=3), lm.LogisticRegression(class_weight='balanced'), lm.LinearRegression()
models = [GaussianNB()]


def score_class_models(models=models):
    acc_score_list = []
    f1_score_list = []

    for model in models:
        print(model)    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_score_list.append(model.score(X_test, y_test))
        f1_score_list.append(f1_score(y_test, y_pred))
        
    for model, score in zip(models, acc_score_list):
        print (f'{model} accuracy: {score * 100} %')
        
    for model, score in zip(models, f1_score_list):
        print(f'{model} f1: {score * 100} %')

    return














# precision_score = []
# recall_score = []
# f1_score = []

# precision_score.append(precision_score(y_test, y_predict))
# recall_score.append(recall_score(y_test, y_predict))
# f1_scores = f1_score(y_true, y_pred, average=None)



'''Cross Validation'''




'''Unsupervised'''
# term frequency matrix has thousands of features.  reduce features using pca?
# k means clustering to find clusters, evaluate clusters, 


kmeans = KMeans(n_clusters=3, n_jobs=-1)
kmeans.fit(count_vec_tfm)


# pca = PCA(n_components=2).fit(count_vec_tfm)