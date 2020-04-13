import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)



train  = pd.read_csv('train_tweets.csv')
test = pd.read_csv('test_tweets.csv')


print(train[train['label'] == 0].head(10))
