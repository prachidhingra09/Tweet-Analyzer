#import matplotlib.pyplot as plt 
#import seaborn as sns
#mport string
#import warnings 
#warnings.filterwarnings("ignore", category=DeprecationWarning)

import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import nltk #for text manipulation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



train  = pd.read_csv('train_tweets.csv')
test = pd.read_csv('test_tweets.csv')


#print(train[train['label'] == 0].head(10))
print(train.shape)
print(test.shape)

#combi = train.append(test, ignore_index=True)
data = train.append(test, ignore_index=True)
print(data.shape)


def delete_twitter_handle(tweet,handle):  # removing pattern of the type @user
  
    r = re.findall(handle,tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)       
  
    return tweet


def tokenization(data): #generating tokens from tweets
    
    tokens = []
    for i in range(len(data)):
        tokens.append(word_tokenize(data['clean_tweets'][i]))
        
    return tokens


def remove_stopwords(tokens): #removing words like is,am,are and special characters
    
    stopwordsList = stopwords.words("english")  
    stopwordsList.extend([',','.','-','!','#'])
    wordsList = []
    for tokenList in tokens:
        words = []
        for word in tokenList:
            if word.lower() not in stopwordsList:
                words.append(word.lower())
        wordsList.append(words)
        
    return wordsList

    
def lemmatization(wordsList): #identifying words or lemma
    
    wnet = WordNetLemmatizer()
    for i in range(len(wordsList)):
        for j in range(len(wordsList[i])):
            wordsList[i][j] = wnet.lemmatize(wordsList[i][j], pos='v')
            
    return wordsList


def vector_transformation(wordsList):
    
    cv = CountVectorizer()
    wordsList = np.asarray(wordsList)
    for i in range(len(wordsList)):
        wordsList[i] = ' '.join(wordsList[i])
    vect = cv.fit_transform(wordsList)
    
    return vect

def train_test_splitting(vect):
    
    y = data['label'].values
    x_train,x_test,y_train,y_test = train_test_split(vect,y,test_size=0.25)
    
    return x_train,x_test,y_train,y_test


def logistic_regression(x_train,x_test,y_train,y_test):
    
    reg = LogisticRegression()
    reg.fit(x_train,y_train)
    y_pred = reg.predict(x_test)
    
    return y_pred


data['clean_tweets'] = np.vectorize(delete_twitter_handle)(data['tweet'], "@[\w]*") 
#print(data.head())

print('\n-@twitter_handle removed-\n')
print(data['clean_tweets'][0])

tokens = tokenization(data)
print('\n-Tokenization-\n')
print(tokens[0])

wordsList = remove_stopwords(tokens)
print('\n-stop words removed-\n')
print(wordsList[0])

wordsList = lemmatization(wordsList)
print('\n-Lemmatization-\n')
print(wordsList[:3])

vect = vector_transformation(wordsList)
x_train,x_test,y_train,y_test = train_test_splitting(vect)
#y_pred = logistic_regression(x_train,x_test,y_train,y_test)
#accuracy = accuracy_score(y_test,y_pred)
#print('\nAccuracy Score-\n')
#print(accuracy)

