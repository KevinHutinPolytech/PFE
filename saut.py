##################################BIBLIOTEQUE################################
# coding: utf8
from __future__ import unicode_literals
import operator 
import json
import sys
import random
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
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
import os
from collections import defaultdict
from pprint import pprint  # pretty-printer
from gensim import models ,corpora, similarities
import tweepy
import time
from tweepy import Stream
from tweepy.streaming import StreamListener

########################################### CONNECTION A API TWITER ###########################################
consumer_key = 'STuwMlRcOAM4x11tvFnhrNfov'
consumer_secret = 'Ow12PHjNB5IkErNB6PrIaYynqIwk9Z4XkRCRlcmXqaLlA19NUd'
access_token = '828603299993645057-pTEQy5rv2ZnOSjvTnnDLLED4KRPkEb7'
access_secret = 'XX4GnbeXW9RSfNjqyDYu9hXkr1ZgKojEaCu0BUMPQh6To'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.secure = True
auth.set_access_token(access_token, access_secret) 
api = tweepy.API(auth)
print("Connecté à l'API Twitter. \n", "Auth : ",auth, " \n API : ", api)

#############################FONCTIONS####################################

class MyListener(StreamListener):

    def __init__(self, fname):
        safe_fname = format_filename(fname)
        self.outfile = "%s.json" % safe_fname
        self.count = 0
    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:                            
                f.write(data)                
                f.close()
                #return True
        except BaseException as e:
            print("quot;Error on_data: %s&quot;" % str(e))
            return True
        
    def on_error(self, status):
        print(status)
        return True
    
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
    
def tracker(query):
    query_fname = ' '.join(query) # string
    twitter_stream = Stream(auth, MyListener(query_fname))
    twitter_stream.filter(track=query, async=True)
    
def getTweetText(tweet):
        
    try:
        tweet = json.loads(tweet)
        #print(type(tweet))
        text = json.dumps(tweet['text'],ensure_ascii = False) # récupere le texte du tweet
        return text
    except:
        print("Probleme dans la récupération du texte")
        print("Tweet : ",tweet)
        pass

# Trait un texte en entrer (unicode) et retourne le texte tokeniser (mode = t),  stemmer (mode = s)

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
     
        #r'(?:\D)', # no numbers
        r"(?:[a-z][a-z\-_]+[a-z])", # words with - and 
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
        terms_stop = []
        for term in tokens:
            if term not in stop:
                try:
                    int(term)
                except:
                    terms_stop.append(term)
        #terms_stop = [term for term in tokens if term not in stop] # Crée une liste avec tout les termes sauf les termes stopé
        if mode == 't' :
            return terms_stop
        if mode == 's' :
            terms_stem = [stemmer.stem(term) for term in terms_stop ]
            return terms_stem
    except:
        print("Problème dans la tokenisation du text")
        print("texte : ",text, "Type : ", type(text), "Mode : ",mode)
        pass
def foundCandidat(tokens):
    listofcandidat = []
    melanchon = ["jlm","jean-luc","melanchon","#jlm","jlm2017"]
    hamon = ["hamon","francois"]
    lepen = ["marine","lepen","pen"]
    fillon = ["francois","fillon"]
    macron = ["macron","emmanuelle"]
    for token in tokens :
        print(token)
        if token in melanchon :
            listofcandidat.append("melanchon")
            print("melanchon")
        if token in hamon :
            listofcandidat.append("hamon")
            print("hamon")
        if token in lepen :
            listofcandidat.append("lepen")
            print("lepen")
        if token in fillon :
            listofcandidat.append("fillon")
            print("fillon")
        if token in macron :
            listofcandidat.append("macron")
            print("macron")
    return listofcandidat
          
            
            
    
######################################## MAIN ################################


mode = sys.argv[1]

if mode == '-h':    
    print('Passez en argument les mots à tracker')
else :   
    query = sys.argv[1:]
    words = [word for word in query]
    tracker(words)
    
    query_fname = ' '.join(query) # string
    safe_fname = format_filename(query_fname)
    filename = "%s.json" % safe_fname
        
    with open(filename,'r',encoding='utf-8',errors='replace') as f:    
        for line in f:        
            m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
            if m != None :
                dico = {}
                text = getTweetText(line)#.decode('unicode-escape')
                dico["text"] = text 
                print("Text : ",text)
                tokens = text2tokens(text,"t")
                print("Tokens: \n",tokens)
                tokens = text2tokens(text,"s")
                print("Stems: \n",tokens)
                print("\n")
                # Fonction qui retourne la liste des candiadats mentionné avec une liste de tokens en entrée 
                listofcandidat = foundCandidat(tokens)
                dico["candidats"] = listofcandidat
                print("Candidats",listofcandidat)
                print("\n")
                #comparer avec classifier eco emploi ...
                # ajouter dico["labels"] = [label max prob 1 , label max prob 2 , label max prob 3]
                
                #comparer avec classifier positif negatif
                # ajouter dico["sentiment"] = sentiment
                
                # ajouter dico dans redis 
                
                
                
                
