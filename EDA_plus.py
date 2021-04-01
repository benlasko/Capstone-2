import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model as lm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


'''
EDA
data_overview:  (multiple EDA functions, info, describe, etc.)
corrs: (correlation table and heatmap)
eda_vis:  (incomplete, visualizations of distributions, correlations)



DATA WRANGLING
drop duplicates:  drops duplicate rows
replace nans/nulls:  replaces nans with mean/median/mode for floats/ints/ojbects/bools
standardize/normalize:  standardizes numerical column data
clean_data combines the above functions


MODEL TESTING AND CROSS VALIDATION
train test split code
a list of models to test and score
model_test:  tests each model in the list of models and return accuracy scores


'''



'''EDA'''


def data_overview(df):
    '''
    Initial EDA on data.

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
        Number of duplicate rows
        Total of null values per column
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
    # print('\u0332'.join("TOTAL DUPLICATE ROWS"))
    # print(f' { int(sum(df[df.duplicated()].sum()))} \n\n')


    return


def corrs(df, cols, corr_round=2):
    '''
    Prints correlation matrix and heatmap for chosen columns of dataframe.
    
    Parameters
    ----------
    df: Pandas dataframe
    
    cols: list  
        List of dataframe columns to find correlations for.  
    
    corr_round: int
        Number of decimals to round correlation values to
    
    Returns
    ----------
    Correlation Matrix: dataframe
        Matrix of correlations

    Heatmap:  seaborn heatmap
        Heatmap showing correlations, annotated with correlation values in percentages
    '''

    df1 = df.copy()
    df1 = df[cols]
    corrs = df1.corr().round(corr_round)

    fig, ax = plt.subplots(figsize=(15,10))
    sns.heatmap(df.corr(), cmap='coolwarm', robust=True, annot=True, fmt='.0%')
    return corrs



def eda_vis(df, cols):
    '''
    EDA visualizations

    Boxplot, scatter plots, histograms to visualize the data.

    Parameters:
    df (pd.dataframe):  A Pandas DataFrame object
    cols (list):  List of columns to see visualizations for.  Default=all columns.

    Returns:
    box_plot
    scatter_matrix

    '''
    # box_plot = df.boxplot[cols]
    # scatter_matrix = pd.plotting.scatter_matrix(df)
    
    # df_num = df.select_dtypes(include=['float64', 'int64'])
    # print(df_num.hist(bins=50)
    return None



def word_count_dict(text):
    d = {}

    for string in example_lst:
        if string in d.keys():
            d[string] += 1
    else:
        d[string] = 1
    return d




    

'''DATA WRANGLING'''

# drop duplicates
# df.drop_duplicates(inplace=True)


# replace nans
def nan_replacer(df, type_='mean'):
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            if type_== 'median':
                median = np.median(df[column].dropna())
                df[column] = df[column].replace(to_replace = np.nan, value = median)

            if type_=='mean':
                mean = np.mean(df[column].dropna())
                df[column] = df[column].replace(to_replace = np.nan, value = mean)
        
        if df[column].dtype in ['bool', 'object']:
            mode = df[column].mode().dropna()
            df['column'] = df['column'].fillna(df['column'].mode()[0])

        else:
            df[column] = df[column].replace(to_replace = np.nan, value = 'None or Unspecified')
    



# Standardizing
def scale_data(data, scaler=StandardScaler()):
    scaler.fit_transform(data)
    return pd.DataFrame(data)


# Normalizing
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals





# Cleaning data
def clean_data(df):
    df1 = df.copy()
    scaler = StandardScaler()

    # drop duplicates
    df1.drop_duplicates(inplace=True)

    # Replace nans with mean in numerical columns
    for col in df1.columns:
        if df1[col].dtype in ['float64', 'int64']:
            mean = np.mean(df1[col].dropna())
            df1[col] = df1[col].replace(to_replace = np.nan, value = mean)


    # Standardize numerical cols
    for col in df1.columns:
        if df1[col].dtypes in ['float64', 'int']:
            scaler.fit(df1)
            data = scaler.transform(df1)


    return df1















'''MODEL TESTING'''
# After cleaning data and deciding which columns to use for predictions on the target column
# Make list of models to use/compare
# Perform train test split on data (split the target data (data to predict based on other data) from the rest of the data.  Make a training set and testing set.
# Get the predictions made by each model
# Find the accuracy scores for the models predictions on the test set to see how well the model will predict new data.
# Print confusion matrices and f1 scores for each model





# 
# Use sklearn's train_test_split to split data into train and test set (X is features, y is targets)
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# Make list of models to use/compare
models=[lm.LinearRegression(), lm.LogisticRegression, KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 275, max_features=5), SVC(), GaussianNB]

# Test and score each model.  Compare and find best accuracy score.
# def model_test(X_train,y_train, X_test, y_test, models:
#     '''
#     Tests and scores multiple models on training data.

#     Params
#         X_train (np.array):  X values in training set from train/test split
#         y_train (np.array):  Y values in training set from train/test split
#         models (list): List of models to find accuracy scores for.

#     Returns
#         Labled scores for each model tested.  '''
    
#     lst = []

#     for model in models:
#         model.fit(X_train, y_train)
#         score_list.append(model.score(X_test, y_test))

#     for model, score in zip(model_types, score_list):
#         print (f'{model}: {score}')

#     return





# A RANDOM FOREST MODEL START TO FINISH WITH CONFUSION MATRIX AND ACCURACY, PRECISION, AND RECALL SCORES

# 1 Remove the features which aren't continuous or boolean
# drop_cols = ['col1', 'col2', 'col3']
# df.drop(columns=drop_cols, inplace=True)



# 2. Make a numpy array called y containing the target values (the column you want to be able to predict based on the feature columns)
# y = df['target_col_name']



# 3. Make a 2 dimensional numpy array containing the feature data (everything except the labels) by making an new df with the target col dropped.
# X = df.drop(columns=['target_col_name'])


# 4. Use sklearn's train_test_split to split into train and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y)


# 5.  Use sklearn's RandomForestClassifier to build a model of your data (fit the model)
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train) 




# 6. What is the accuracy score on the test data?
# print("\n8. score:", rf.score(X_test, y_test))


# 7. Draw a confusion matrix for the results
# y_predict = rf.predict(X_test)
# print("\n9. confusion matrix:")
# print(confusion_matrix(y_test, y_predict))

# 8. What is the precision? Recall?
# print("\n10. precision:", precision_score(y_test, y_predict))
# print("    recall:", recall_score(y_test, y_predict))

# 8.5  What is the f1 score?
# print(f'F1 Score: ', f1_score(y_true, y_predict, average=None))


#  9. Build the RandomForestClassifier again setting the out of bag parameter to be true.  Compare the score to see if out of bag is better.
# rf = RandomForestClassifier(n_estimators=30, oob_score=True)
# rf.fit(X_train, y_train)
# print("\n11: accuracy score:", rf.score(X_test, y_test))
# print("    out of bag score:", rf.oob_score_)


# 10. Use sklearn's model to get the feature importances
# feature_importances = np.argsort(rf.feature_importances_)
# print("\n12: top five:", list(df.columns[feature_importances[-1:-6:-1]]))
## top five: ['Day Charge', 'Day Mins', 'CustServ Calls', "Int'l Plan", 'Eve Charge']
## (will vary a little)


# 10.5  Also See permutation importance for the model and plot them.  Function for that:

def plot_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx[:10]].T,
               vert=False, labels=sorted_idx)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()













# def decision_tree(df):
#     """
#     Applies a decision tree model to the DataFrame.
#     Input: DataFrame
#     Output: Accuracy Score, Confusion Matrix, visualization of tree
#     """
#     X_train, X_test, y_train, y_test, X, y = split(df)
#     clf = DecisionTreeClassifier()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
#     print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))


# def logistic_regression(df):
#     """
#     Performs Logistic Regression on a passed in data set.
#     Calls the split function to split data then fits split data to model.
#     **Important** to note that this logistic regression takes in a PCA DataFrame.
#     Input: DataFrame
#     Output: Coefficients of Logistic regression, Confusion Matrix, Model score
#     """
#     X_train, X_test, y_train, y_test, X, y = split(df)
#     log_reg = LogisticRegression()
#     log_reg.fit(X_train, y_train)
#     y_pred = log_reg.predict(X_test)
#     print("Coefficients:",log_reg.coef_)  # determine most important questions
#     print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
#     print('Logistic Regression Accuracy: ', log_reg.score(X, y))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))

# def random_forest(df):
#     """
#     Performs Logistic Regression on a passed in data set.
#     Calls the split function to split data then fits split data to model.
#     **Important** to note that this logistic regression takes in a PCA DataFrame.
#     Input: DataFrame
#     Output: Coefficients of Logistic regression, Confusion Matrix, Model score
#     """
#     X_train, X_test, y_train, y_test, X, y = split(df)
#     forest = RandomForestClassifier(n_estimators=10, criterion='entropy')
#     forest.fit(X_train, y_train)
#     y_pred = forest.predict(X_test)
#     print("Coefficients:",log_reg.coef_)  # determine most important questions
#     print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
#     print('Logistic Regression Accuracy: ', log_reg.score(X, y))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))



# def model_scores(X_train, y_train):
#     # Logistic Regression
#     log = LogisticRegression()
#     log.fit(X_train,y_train)

#     # K Nearest Neighbors
#     knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', =2)
#     knn.fit(X_train,y_train)

#     # Decision Tree
#     tree = DecisionTreeClassifier(criterion='entropy')
#     tree.fit(X_train,y_train)

#     # Random Forest
#     forest = RandomForestClassifier(n_estimators=10, criterion='entropy')
#     forest.fit(X_train,y_train)

#     # Support Vector Machine Linear 
#     svc_lin = SVC(kernel='linear')
#     svc.fit(X_train,y_train)

#      # Support Vector Machine rbf 
#     svc_rbf = SVC(kernel='rbf')
#     svc_rbf.fit(X_train,y_train)

#     # GaussianNB
#     gauss = GaussianNB()
#     gauss.fit(X_train,y_train)

#     print("\u0332".join("Accuracy Scores "))
#     print('Logistic Regression: ', log.score(X_train,y_train))
#     print('KNN: ', knn.score(X_train,y_train))
#     print('Decision Tree: ', tree.score(X_train,y_train))
#     print('Random Forest: ', forest.score(X_train,y_train))
#     print('SVC Linear: ', svc_lin.score(X_train,y_train))
#     print('SVC rbf: ', svc_rbf.score(X_train,y_train))
#     print('Naive Bayes: ', gauss.score(X_train,y_train))

#     return log, knn, tree, forest, svc_lin, svc_rbf, guass






# Show confusion matrix for each model and Show F1 score for each model.
def conf_mat_f1_score(models):

    for model in models:
        print(f'{model} confusion matrix: ', confusion(y_true, y_pred, average=None))
        print(f'{model} F1 Score: ', f1_score(y_true, y_pred, average=None))
        print()









# See permutation importance for final model

def plot_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx[:10]].T,
               vert=False, labels=sorted_idx)
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()