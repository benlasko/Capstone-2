# **Sentiment Analysis on Tweets**
## Capstone 2 for the Galvanize Data Science Immersive
### Natural language processing is a profound tool with the power to help us understand our collective history, culture, and state of mind.  One way to apply it is conducting sentiment analysis which can be used for a variety of purposes including assessing public sentiment about products, events, or ideas, predicting social unrest, informing education and policy, and helping decide how to allocate resources.  So although the dataset I used for this project was a collection of tweets about airlines my interest in doing this was to learn and practice using natural language processing and predictive modeling for sentiment analysis to be able to apply these techniques in a variety of different ways.  To do this I wrote code and built functions to perform EDA, text cleaning, text analysis, predictive modeling, cross validation, and supervised/unsupervised learning techniques.


###  The dataset I used had a collection of tweets from February of 2015 about six major airlines.  The tweets were prelabeled by sentiment.  I started by putting the data into a Pandas dataframe, replacing NaNs, removing duplicate rows, and shortening timeseries data to the day and converting it to datetime objects, although I mainly used the tweets and their prelabeled sentiments in my analysis. 


###  The tweets were classified into positive, negative, and neutral sentiments. The distribution was 63% negative, 21% neutral, and 16% positive.

<br>

![Sentiment-Counts](/images/sent-counts.png)

<br>

![Sentiment-Counts-Airline](/images/sent-counts-airline.png)


### This wordcloud shows the most common words from all the tweets before the text was cleaned and analyzed.

![Unclean-Corp-WC](/images/unclean-corp-wc.png)

<br>

### To be able to analyze and process the tweets I made a text cleaning pipeline that does the following:
* Lowercases text
* Removes puncuation
* Removes numbers
* Removes new lines
* Removes urls
* Splits text to a list of words
* Removes stopwords
* Lemmatizes the words
* Joins list of words back to a string
* Returns the cleaned text

### So text like this:
1. @VirginAmerica why are your first fares in May over three times more than other carriers when all seats are available to select???
2. In his house at R'lyeh dead Cthulhu waits dreaming. 
3. To be or not to be?
4. @USAirways 2 hours and counting waiting to get into a gate in Philadelphia. Just icing on the cake for a miserable flight experience


### Becomes:
1. first fare may time carrier seat available select
2. house rlyeh dead cthulhu wait dreaming 
3. ''
4. hour counting waiting get gate philadelphia icing cake miserable experience

### Line 2 is a good example of why natural language processing can be difficult.

<br>

### The cleaned corpus had about 18000 unique words and the lexical diversity of the corpus was somewhat limited.
 
 * Total words: 129374
 * Total unique words: 18350
 * Proportion of unique words: 0.14

### The wordcloud and graph below show the most common words from all the tweets after going through the cleaning process.


![Clean-WC](/images/clean-corp-wc.png)

![Most-Common-Words](/images/common-words.png)

<br>

### Examples of the context in which some the most common words were found.
#### "thank"

![thank-context](/images/thank-context.png)

#### "service"

![service-context](/images/service-context.png)

<br>

### I used supervised learning classification models to try and build a good predictive model to classify tweets into their positive, negative, or neutral categories.  Since tweets are short I used Sklearn's CountVectorizer instead of the tfidf vectorizer. 

* Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine
* Multi Layer Perceptron neural network

### Using random guessing the chances of correctly classifying a tweet would be

### Running models with the default parameters resulted in accuracy and f1 scores between 70-78% with Support Vector Machine performing the best.  F1 scores were usually only about 1-2% less than accuracy scores.

#### Model scores using default parameters.

![No-Tuning](/images/no-tuning.png)

### I increased the number of trees in the Random Forest model as high as 12000, and tried different max features and class weights but didn't get any scores higher than 77%.

### For the Multi Layer Perceptron neural network I added up to 500 layers, experimented with alphas from .0001 to .05, used adaptive and invscaling learning rates, tried different solvers, tried relu and tanh activations, and batch sizes as low as 10.  I wasn't able to tune the MLP model to get a score higher than about 77%.

### I also tried tuning the SVM SVC using different parameters for regularization strength (C=3, C=6) and with "balanced" for the class_weight parameter.

### By tuning the models to account for the imbalanced classes and making changes to my stopwords list and text pipeline I was able to raise the MLP classifier accuracy about 7% but other models performed relatively similarly to when they ran using the default parameters.  SVM, Random Forest, and MLP models performed better than the others I tried.

#### Model scores after tuning.

![Some-Tuning](/images/some-tuning.png)


### The model with the highest score, 78.15%, was the Support Vector Machines SVC(C=3) model.  Here's the confusion matrix for that model.I used stratifed KFold cross validation to test the model and the results were  .


### Finally, although the dataset had the tweets prelabeled, I wanted to try using unsupervised learning techniques to analyze the corpus.  I used a tfidf vectorizer to create a term frequency matrix of the entire corpus, used PCA to shrink the matrix from over 13000 features to 100 features, and used KMeans Clustering to form three clusters.

<br>

### In the future I'd like to revisit this project to better tune the supervised learning models, include n-grams in my text analysis, and analyze the clusters formed by KMeans to create labels for each cluster.


![Some-Tuning](/images/maybe-just-complain.png)