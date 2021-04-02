# **Sentiment Analysis on Tweets**
### Sentiment analysis can be used for a variety of purposes including assessing public sentiment about products or events, predicting social unrest, guiding policy making, and helping to decide how to best allocate resources.  So although the dataset I used for this project has about 14000 tweets about major airlines, each categorized as positive, negative, or neutral, my interest in doing this was to learn and practice using Natural Language Processing and predictive modeling for sentiment analysis to be able to apply it in a variety of different ways.  To do this I built functions to perform EDA, text analysis, text cleaning, predictive modeling, cross validation, and unsupervised learning techniques.

<br>

###  I started by getting the data into a Pandas dataframe, replacing NaNs, removing duplicate rows, and shortening timeseries data to the day and into datetime objects, although I mainly used the tweets and their prelableled sentiments in my analysis.  Tweets were categorized as positive, negative, and neutral.  

<br>

###  The tweets were mostly negative which was no surprise with Twitter.  The distribution of the tweets was about 60% negative, 25% neutral, and 15% positive.

<br>


![Sentiment-Counts](/images/sent-counts.png)

<br>


![Sentiment-Counts-Airline](/images/sent-counts-airline.png)


<br>


### This wordcloud shows the most common words from all the tweets before the text was processed.

![Unclean-Corp-WC](/images/unclean-corp-wc.png)


<br>

### I made a text processing pipeline that did the following:
* Lowercased text
* Removed puncuation
* Removed numbers
* Removed new lines
* Removed urls
* Split text to a list of words
* Removed stopwords
* Lemmatized the text
* Joined word list back to a string
* Returned the cleaned text

### Before cleaning:
* 
* 
* 


<br>

### After cleaning:
* 
* 
* 

<br>

### The wordcloud and graph below show the most common words from all the tweets after going through the cleaning process.

<br>

![Clean-WC](/images/clean-corp-wc.png)

![Most-Common-Words](/images/common-words.png)

<br>

### The cleaned corpus had about 18000 unique words and the lexical diversity of the corpus was somewhat limited.
 
 * Total words: 129374
 * Total unique words: 18350
 * Average word repetition: 36289.93
 * Proportion of unique words: 0.14

<br>

### I used some supervised learning classification models to try and find a good predictive model to categorize tweets into positive, negative, or neutral categories.  For each model I found the accuracy and F1 scores.

* Naive Bayes
* Decision Tree
* Random Forest
* Support Vector Machine
* Multi Layer Perceptron neural network

<br>

### Using random guessing the chances of correctly classifying a tweet would be

<br>

### Running models with the default parameters resulted in accuracy and f1 scores between 70% - 78% with the Support Vector Machine performing the best.

### Model scores using default parameters.
![No-Tuning](/images/no-tuning.png)

### I increased the number of trees in the Random Forest model as high as 12000, and tried different max features and class weights but didn't get any scores higher than 77%.

<br>

### For the Multi Layer Perceptron neural network I added up to 500 layers, experimented with alphas from .00001 to .05, used adaptive and inversely scaled learning rates, tried different solvers, batch sizes as low as 10, and tanh activation.  I wasn't able to tune the MLP model to get a score higher than about 77%.

<br>

### By tuning the models to account for the imbalanced classes and making changes to my stopwords list I was able to raise the Multi Layer Perceptron neural network model accuracy about 7% from its original scores around 70% but other models performed relatively simliarly to running them with the default parameters.  SVM, Random Forest, and Multi Layer Perceptron models performed the best.

<br>

### Model scores after tuning.
![Some-Tuning](/images/some-tuning.png)

<br>

### The model with the highest score was  .   I used stratifed K Fold cross validation to test the model and the results were  .


<br>


### Finally, although the dataset had the tweets prelabeled, I wanted to try using unsupervised learning techniques to analyze the corpus.  I used a tfidf vectorizer to create a term frequency matrix, used PCA to shrink the matrix from over 13000 features to 100, and used KMeans Clustering to form three clusters.

<br>

### In the future I'd like to revisit this project to better tune the supervised learning models and analyze the clusters formed by KMeans to find their simliarities and be able to create labels for each cluster.