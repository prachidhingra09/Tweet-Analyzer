import utils
import random
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer

''' Built by: iimashfaaq & prachidhingra09 '''

# FILES
TRAIN_FILE  = pd.read_csv('./datasets/train_tweets.csv')
TEST_FILE = pd.read_csv('./datasets/test_tweets.csv')
TRAIN_PROCESSED_FILE = './datasets/TRAIN_PROCESSED_FILE.csv'
TEST_PROCESSED_FILE = './datasets/TEST_PROCESSED_FILE.csv'

# TRUE WHILE TRAINING
TRAIN = False
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


# DECISION TREE FUNCTIONS
def get_feature_vector(tweet):
    uni_feature_vector = []
    bi_feature_vector = []
    words = tweet.split()
    for i in range(len(words) - 1):
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
    for i in range(num_batches):
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
    """ Fits X for TF-IDF vectorization and returns the transformer """
    transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
    transformer.fit(X)
    return transformer


def process_tweets(csv_file, test_file=True):
    """ Returns a list of tuples of type (tweet_id, feature_vector)
            or (tweet_id, sentiment, feature_vector) """
    tweets = []
    print('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id = line[: line.find(',')]
                tweet = line[1 + line.find(','):]
            else:
                tweet_id = line[: line.find(',')]
                sentiment = int(line[:line.find(',')])
                tweet = line[1 + line.find(','):]
                # tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append((tweet_id, feature_vector))
            else:
                tweets.append((tweet_id, int(sentiment), feature_vector))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets


def save_processed_file(linesList, processed_file_name, csv_file_name, test_file=False):
    ''' SAVING THE PROCESSED DATA INTO
        A processed.csv FILE FOR FURTHER 
        CLASSIFICATION '''
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()[1:]
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
    print('Saved processed tweets to: %s' % processed_file_name)
    return processed_file_name

if __name__ == "__main__":

    # print(TRAIN_FILE.shape)
    # print(TEST_FILE.shape)

    data = TRAIN_FILE.append(TEST_FILE, ignore_index=True)
    # print(data.shape)


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

    for i in range(len(wordsList)):
        wordsList[i] = ' '.join(wordsList[i])
    
    print(wordsList[:3])

    # FUNCTION CALL TO SAVE THE PROCESSED DATA INTO CSV FILES
    save_processed_file(wordsList[:31963], './datasets/TRAIN_PROCESSED_FILE.csv', './datasets/train_tweets.csv', False)
    save_processed_file(wordsList[31962:], './datasets/TEST_PROCESSED_FILE.csv', './datasets/test_tweets.csv', True)

    np.random.seed(1337)
    unigrams = utils.top_n_words('./datasets/train_tweets-freqdist.pkl', UNIGRAM_SIZE)
    if USE_BIGRAMS:
        bigrams = utils.top_n_bigrams('./datasets/train_tweets-freqdist-bi.pkl', BIGRAM_SIZE)

    tweets = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    if TRAIN:
        train_tweets, val_tweets = utils.split_data(tweets)
    else:
        random.shuffle(tweets)
        train_tweets = tweets
    del tweets
    print('Extracting features & training batches')
    clf = DecisionTreeClassifier(max_depth=25)
    batch_size = len(train_tweets)
    i = 1
    n_train_batches = int(np.ceil(len(train_tweets) / float(batch_size)))
    for training_set_X, training_set_y in extract_features(train_tweets, test_file=False, feat_type=FEAT_TYPE, batch_size=batch_size):
        utils.write_status(i, n_train_batches)
        i += 1
        if FEAT_TYPE == 'frequency':
            tfidf = apply_tf_idf(training_set_X)
            training_set_X = tfidf.transform(training_set_X)
        clf.fit(training_set_X, training_set_y)
    print('\n')
    print('Testing')
    if TRAIN:
        # TRAINING PHASE
        correct, total = 0, len(val_tweets)
        i = 1
        batch_size = len(val_tweets)
        n_val_batches = int(np.ceil(len(val_tweets) / float(batch_size)))
        for val_set_X, val_set_y in extract_features(val_tweets, test_file=False, feat_type=FEAT_TYPE, batch_size=batch_size):
            if FEAT_TYPE == 'frequency':
                val_set_X = tfidf.transform(val_set_X)
            prediction = clf.predict(val_set_X)
            correct += np.sum(prediction == val_set_y)
            utils.write_status(i, n_val_batches)
            i += 1
        print('\nCorrect: %d/%d = %.4f %%' % (correct, total, correct * 100. / total))
    else:
        # TESTING PHASE
        del train_tweets
        test_tweets = process_tweets(TEST_PROCESSED_FILE, test_file=True)
        n_test_batches = int(np.ceil(len(test_tweets) / float(batch_size)))
        predictions = np.array([])
        print('Predicting batches')
        i = 1
        for test_set_X, _ in extract_features(test_tweets, test_file=True, feat_type=FEAT_TYPE):
            if FEAT_TYPE == 'frequency':
                test_set_X = tfidf.transform(test_set_X)
            prediction = clf.predict(test_set_X)
            predictions = np.concatenate((predictions, prediction))
            utils.write_status(i, n_test_batches)
            i += 1
        predictions = [(str(j), int(predictions[j])) for j in range(len(test_tweets))]
        utils.save_results_to_csv(predictions, 'decisionTreeResult.csv')
        print('\nSaved to decisionTreeResult.csv\n')

        # PRECISION F1 SCORES

        x_train,x_test,y_train,y_test = train_test_splitting(vect)
        x_train = np.nan_to_num(x_train)
        y_train = np.nan_to_num(y_train)
        x_test = np.nan_to_num(x_test)
        y_test = np.nan_to_num(y_test)
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        print('\nConfusion Matrix - \n')
        print(confusion_matrix(y_test, y_pred))
        print('\nREPORT - \n')
        print(classification_report(y_test, y_pred))
        print('\nAccuracy Score DECISION TREE -')
        accuracy = accuracy_score(y_test,y_pred)
        print(accuracy*100 + '%')
