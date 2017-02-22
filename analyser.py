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
 
from nltk.corpus import stopwords
import string
from nltk import bigrams 


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
   # r'([a-z0-9]+(\\u)[a-z0-9]+)'
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

"""
The regular expressions are compiled with the flags re.VERBOSE, to allow spaces in the regexp to be ignored (see the multi-line emoticons regexp),
and re.IGNORECASE to catch both upper and lowercases. The tokenize() function simply catches all the tokens in a string and returns them as a list.
This function is used within preprocess(), which is used as a pre-processing chain: in this case we simply add a lowercasing feature for all the
tokens that are not emoticons (e.g. :D doesn’t become :d).
"""
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


count = 0
punctuation = list(string.punctuation)
stop = stopwords.words('french') + punctuation + ['via','le','les','a'] # Liste des tokens à effacer

with open('stream__Macron.json', 'r') as f:
    count_all = Counter() # Inisialise un compteur
    count_stop = Counter() # Inisialise un compteur
    count_bi = Counter() # Inisialise un compteur
    count_user_name = Counter() # Inisialise un compteur   
    for line in f:
        m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne
        
        if m != None :
            try  :
                tweet = json.loads(line)
                count = count + 1 # compte le nombre de tweet
                #print(count)
                          
            
                user = json.loads(json.dumps(tweet['user']))
                user_name = json.dumps(user['screen_name'],ensure_ascii = False)
                user_location = json.dumps(user['location'],ensure_ascii = False)
                count_user_name.update(preprocess(user_name))
                date = json.dumps(tweet['created_at'])
                #print(date)
                print(user_name)
                #print(user_location)
                print(json.dumps(tweet['text'],ensure_ascii = False))
           # print('')
            
                tweet = json.dumps(tweet['text'],ensure_ascii = False) # récupere le texte du tweet
                tokens = preprocess(tweet) # Tokenise le texte
                
                terms_all = [term for term in preprocess(tweet)] # Crée une liste avec tout les terms
                terms_stop = [term for term in preprocess(tweet) if term not in stop] # Crée une liste avec tout les termes sauf les termes stopé
                terms_bigram = bigrams(terms_stop) # Compte les termes les plus fréquent deux à deux 
                count_all.update(terms_all) # Met à jour le compteur avec les termes en parametres
                count_stop.update(terms_stop) # Met à jour le compteur avec les termes en parametres
                count_bi.update(terms_bigram) # Met à jour le compteur avec les termes en parametres
            except:
                pass
print(count)
print('stop')
print(count_stop.most_common(40)) # Affiche les 20 mots les plus frequents
print('bigram')
print(count_bi.most_common(40)) # Affiche les 20 mots les plus frequents 
print('')
print(count_user_name.most_common(40))
            
