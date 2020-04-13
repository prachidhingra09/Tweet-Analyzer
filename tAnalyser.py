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


# FILES
TRAIN_FILE  = pd.read_csv('./datasets/train_tweets.csv')
TEST_FILE = pd.read_csv('./datasets/test_tweets.csv')
PROCESSED_FILE = 'processed.csv'

# TRUE WHILE TRAINING
TRAIN = True
UNIGRAM_SIZE = 15000
VOCAB_SIZE = UNIGRAM_SIZE

# IF USING BIGRAMS
USE_BIGRAMS = False
if USE_BIGRAMS:
    BIGRAM_SIZE = 10000
    VOCAB_SIZE = UNIGRAM_SIZE + BIGRAM_SIZE
FEAT_TYPE = 'frequency'

def delete_twitter_handle(tweet, handle):  # removing pattern of the type @user
    
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

# DECISION TREE FUNCTIONS
def get_feature_vector(tweet):
    uni_feature_vector = []
    bi_feature_vector = []
    words = tweet.split()
    for i in xrange(len(words) - 1):
        word = words[i]
        next_word = words[i + 1]
        if unigrams.get(word):
            uni_feature_vector.append(word)
        if USE_BIGRAMS:
            if bigrams.get((word, next_word)):
                bi_feature_vector.append((word, next_word))
    if len(words) >= 1:
        if unigrams.get(words[-1]):
            uni_feature_vector.append(words[-1])
    return uni_feature_vector, bi_feature_vector


def extract_features(tweets, batch_size=500, test_file=True, feat_type='presence'):
    num_batches = int(np.ceil(len(tweets) / float(batch_size)))
    for i in xrange(num_batches):
        batch = tweets[i * batch_size: (i + 1) * batch_size]
        features = lil_matrix((batch_size, VOCAB_SIZE))
        labels = np.zeros(batch_size)
        for j, tweet in enumerate(batch):
            if test_file:
                tweet_words = tweet[1][0]
                tweet_bigrams = tweet[1][1]
            else:
                tweet_words = tweet[2][0]
                tweet_bigrams = tweet[2][1]
                labels[j] = tweet[1]
            if feat_type == 'presence':
                tweet_words = set(tweet_words)
                tweet_bigrams = set(tweet_bigrams)
            for word in tweet_words:
                idx = unigrams.get(word)
                if idx:
                    features[j, idx] += 1
            if USE_BIGRAMS:
                for bigram in tweet_bigrams:
                    idx = bigrams.get(bigram)
                    if idx:
                        features[j, UNIGRAM_SIZE + idx] += 1
        yield features, labels


def apply_tf_idf(X):
    """
    Fits X for TF-IDF vectorization and returns the transformer.
    """
    transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    transformer.fit(X)
    return transformer


def process_tweets(csv_file, test_file=True):
    """Returns a list of tuples of type (tweet_id, feature_vector)
            or (tweet_id, sentiment, feature_vector)

    Args:
        csv_file (str): Name of processed csv file generated by preprocess.py
        test_file (bool, optional): If processing test file

    Returns:
        list: Of tuples
    """
    tweets = []
    print('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets

''' SAVING THE PROCESSED DATA INTO
    A processed.csv FILE FOR FURTHER 
    CLASSIFICATION '''
def save_processed_file(linesList, processed_file_name, csv_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')

    # if not test_file:
    #     csv = open(TRAIN_FILE, 'r')
    # else:
    #     csv = open(TEST_FILE, 'r')

    # total = len(linesList)
    # for i, line in enumerate(linesList):
    #     tweet_id = i
    #     if not test_file:
    #         save_to_file.write('%s,%d,%s\n' % (tweet_id, sentiment, line))
    #     else:
    #         save_to_file.write('%s,%s\n' % (tweet_id, line))
    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                sentiment = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            tweet = line
            if not test_file:
                save_to_file.write('%s,%d,%s\n' % (tweet_id, sentiment, linesList[i]))
            else:
                save_to_file.write('%s,%s\n' % (tweet_id, linesList[i]))
            # write_status(i + 1, total)

    save_to_file.close()
    print('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name

if __name__ == "__main__":

    print(TRAIN_FILE.shape)
    print(TEST_FILE.shape)

    data = TRAIN_FILE.append(TEST_FILE, ignore_index=True)
    print(data.shape)


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

    for i in range(len(wordsList)):
        wordsList[i] = ' '.join(wordsList[i])
    
    print(wordsList[:3])

    save_processed_file(wordsList[:31963], './datasets/TRAIN_PROCESSED_FILE.csv', TRAIN_FILE, False)
    save_processed_file(wordsList[31964:], './datasets/TEST_PROCESSED_FILE.csv', TEST_FILE, True)
    # vect = vector_transformation(wordsList)
    # x_train, x_test, y_train, y_test = train_test_splitting(vect)
    
    #y_pred = logistic_regression(x_train,x_test,y_train,y_test)
    #accuracy = accuracy_score(y_test,y_pred)
    #print('\nAccuracy Score-\n')
    #print(accuracy)
