##################################BIBLIOTEQUE################################
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

#############################FONCTIONS####################################


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
    stop = stopwords.words('french') + punctuation + ['via','le','les','a','rt'] # Liste des tokens à effacer

    stemmer = SnowballStemmer('french')
    tokens = tokens_re.findall(unidecode(text))
    tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    if mode == 't' :
        return tokens
    terms_stop = [term for term in tokens if term not in stop] # Crée une liste avec tout les termes sauf les termes stopé
    if mode == 's' :
        terms_stem = [stemmer.stem(term) for term in terms_stop ]
        return terms_stem




######################################## MAIN ################################

while true :
    print("1 : Tracker des tweet sur twitter ")
    print("2 : Stocker JSON dans redis ")
    print("3 : Recuperer texte d'un JSON provenant de redis ")
    print("4 : tokeniser un text ")
    print('5 : LDA')
    mode = input("Quel mode choisir ? ")
    if mode == 1 :
        print('Non implémenté')
    if mode == 2 :
        print('Non implémenté')
    if mode == 3 :
        print('Non implémenté')
    if mode == 4 :
        print('1 : tokeniser un fichier .txt')
        print('2 : tokeniser un une chaine de caractere')
        sousmode = input("Quel mode choisir ? ") 
        if sousmode == 1 :        
            filename = input("Quel est le nom du fichier ou chemain d'accès ? ")
            try :
                with open(filename,'r') as f:    
                for line in f:        
                    tokens = text2tokens(line.decode('unicode-escape'),s)
                    print(tokens)
            except :
                print(" Erreur 41 ")
                print("le nom du fichier doit être de la forme monfichier.txt ou /sousdossier/monfichier.txt encodé en ANSII")

        if sousmode == 2 :        
            chaine = input("Entrez la chaine de caractère :  ")
            try :                                        
                    tokens = text2tokens(chaine.decode('unicode-escape'),s)
                    print(tokens)
            except :
                print(" Erreur 42 ")
               


if mode == 5 :
    print('Non implémenté')
