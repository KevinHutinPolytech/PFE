# coding: utf8
from __future__ import unicode_literals
####################################################################################
# Traiter les data tokenisé
####################################################################################

#
# Dans un premier temps on remani le tokeniser afin qu'il prenne en compte les emoticons, les RT, les @-mention, les url, ...
#
import operator 
import json
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
import redis
from nltk.corpus import stopwords
import string
from nltk import bigrams 
from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def getTweetsByHash(filename,database): 
    listOfTweets = database.hvals(filename)
    return listOfTweets

def getTweetText(TweetStr):
    tweetDict = json.loads(TweetStr)
    tweetText = unidecode(tweetDict['text'])
    return tweetText

#######TEST####
##database = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
##listOfTweets = getTweetsByHash("presidentielle.json",database)
##tweetText = getTweetText(listOfTweets[0])
##print(tweetText)
###############

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)', # anything else
    
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
stemmer = SnowballStemmer("french")
"""
The regular expressions are compiled with the flags re.VERBOSE, to allow spaces in the regexp to be ignored (see the multi-line emoticons regexp),
and re.IGNORECASE to catch both upper and lowercases. The tokenize() function simply catches all the tokens in a string and returns them as a list.
This function is used within preprocess(), which is used as a pre-processing chain: in this case we simply add a lowercasing feature for all the
tokens that are not emoticons (e.g. :D doesn’t become :d).
"""

database = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
listOfTweets = getTweetsByHash("presidentielle.json",database)

punctuation = list(string.punctuation)
stop = stopwords.words('french') + punctuation + ['via','le','les','a','rt'] # Liste des tokens à effacer

stemmer = SnowballStemmer("french")

print(stemmer)
print(type(stemmer))
count_stop = Counter() # Inisialise un compteur
for tweet in listOfTweets:
    try:
        tweetText = getTweetText(tweet)
        print(tweetText)
        tokens = preprocess(tweetText) # Tokenise le texte
        stem = stemmer.stem(tokens)
        print(stem)
        print(type(stem))
        terms_stop = [term for term in tokens if term not in stop] # Crée une liste avec tout les termes sauf les termes stopé
        stem = stemmer.stem(terms_stop)
        print(stem)
        print(type(stem))
        count_stop.update(stem) # Met à jour le compteur avec les termes en parametres
        
    except:
        tweetText = getTweetText(tweet)
        print(tweetText)
        tokens = preprocess(tweetText) # Tokenise le texte
        stem = stemmer.stem(tokens)
        print(stem)
        print(type(stem))
        print('fail')
        pass

print(count_stop.most_common(40)) # Affiche les 20 mots les plus frequents
