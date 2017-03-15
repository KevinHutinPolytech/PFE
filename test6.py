# coding: utf8
from __future__ import unicode_literals
import operator 
import json
import sys
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
import os
from collections import defaultdict
from pprint import pprint  # pretty-printer
from gensim import models ,corpora, similarities
import tweepy
import time
from tweepy import Stream
from tweepy.streaming import StreamListener

def format_filename(fname):
    """Convert fname into a safe string for a file name.
    Return: string
    """
    return ''.join(convert_valid(one_char) for one_char in fname)

def convert_valid(one_char):
    """Convert a character into '' if "invalid".
    Return: string
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return ''

def getTweetText(tweet):
        
    try:
        #print('getTweetText')
        #print(type(tweet))
        #print(tweet)
        tweet = json.loads(tweet)
        #print(type(tweet))
        text = json.dumps(tweet['text'],ensure_ascii = False) # récupere le texte du tweet
        return text
    except:
        print("Probleme dans la récupération du texte")
        print("Tweet : ",tweet)
        pass

def text2tokens(text,mode):
    
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
    """
    The regular expressions are compiled with the flags re.VERBOSE, to allow spaces in the regexp to be ignored (see the multi-line emoticons regexp),
    and re.IGNORECASE to catch both upper and lowercases. The tokenize() function simply catches all the tokens in a string and returns them as a list.
    This function is used within preprocess(), which is used as a pre-processing chain: in this case we simply add a lowercasing feature for all the
    tokens that are not emoticons (e.g. :D doesn’t become :d).
    """
    punctuation = list(string.punctuation)
    stop = stopwords.words('french') + punctuation + ['>>','<<','<','>','via','le','les','a','rt'] # Liste des tokens à effacer

    stemmer = SnowballStemmer('french')
    try:
        tokens = tokens_re.findall(unidecode(text))
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]    
        terms_stop = [term for term in tokens if term not in stop] # Crée une liste avec tout les termes sauf les termes stopé
        if mode == 't' :
            return terms_stop
        if mode == 's' :
            terms_stem = [stemmer.stem(term) for term in terms_stop ]
            return terms_stem
    except:
        print("Problème dans la tokenisation du text")
        print("texte : ",text, "Type : ", type(text), "Mode : ",mode)
        pass

wordkey = input("Entrer la chaine a tracker : ")
topic = input("Entrer le topic dans lequel s'inscrit ce mot cle : ")
#tracktweet
query_fname = ' '.join(wordkey) # string
safe_fname = format_filename(query_fname)
filename = "%s.json" % safe_fname
list_dico =[]
count = 0
with open(filename,'r') as f:    
    for line in f:        
        count = count +1
        text = getTweetText(line)
        print(text)
        tokens = text2tokens(text,"t")
        stems = text2tokens(text,"s")
        dico = {}
        dico["id_tweet"] = count
        dico["tokens"] = tokens
        dico["stems"] = stems
        dico["topic"] = topic
        list_dico.append(dico)
print(list_dico)
