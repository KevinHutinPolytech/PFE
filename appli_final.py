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
    
def countTweetInJson(filename):
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
            if m != None :
                count = count + 1              
        f.close()
    return count

def json2redis(filename,database):
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
            if m != None :
                try :                
                    tweet = json.loads(line)
                    date = json.dumps(tweet['created_at'])
                    database.hset(filename,date,tweet)
                    print('Importation réussi')
                except:
                    print('failed try push json into redis')
                    print("Nom du fichier : ",filename," Tweet : ", line, " Type tweet : " , type(line))
                    pass

# Fonctionne lorsque on utilise un fichier json mais pose des problème quand onextrait un json provenant d'un redis car convertit en str
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

def redis2json(hashname, database):
    jsonfile = database.hvals(hashname)
    return jsonfile

def json2tweet(line):
    print('json2tweet')
    tweet = json.loads(line)
    return tweet

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
def txt2lda(monfichier):    
    
    with open(monfichier,'r') as f:    
        texts = [[tokens for tokens in text2tokens(line.decode('unicode-escape'),"s") if len(tokens) != 0 ] for line in f ]
        # remove words that appear only once            
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(texts)
        dictionary.save('/tmp/emploi.dict')  # store the dictionary, for future reference
        print(dictionary)
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('/tmp/emploi.mm', corpus)  # store to disk, for later use
        print(corpus)
        print(len(texts))
        
        lda = models.LdaModel(corpus, id2word=dictionary, num_topics=len(texts))
        print("Génération d'un model LDA...")
        pprint(lda)
        print("LDA généré")
        
        doc = " Le marche de l'emploi est en chute libre, le nombre de chomeur ne cesse d'augmenter "
        doc_bow = dictionary.doc2bow(text2tokens(doc.decode('unicode-escape'),"s"))
        print(lda[doc_bow]) # get topic probability distribution for a document
        vec_lda = lda[doc_bow]
        for i in vec_lda : 
            topicid = i[0]
            print(i[0])
            print(lda.get_topic_terms(topicid, topn=10))
        
        f.close()
    return lda
       
        
    
       


######################################## MAIN ################################

while True :
    print("1 : Tracker des tweet sur twitter ")
    print("2 : Stocker JSON dans redis ")
    print("3 : Recuperer texte d'un JSON provenant de redis ")
    print("4 : tokeniser un text ")
    print('5 : Créer un corpus avec le model LDA')    
    print('6 : Track une chaine dans tweeter et donne une liste de dict{id_tweet : , tokens: , stems: , topic: }')
    print('7 : Compte le nombre de tweet dans un fichier json')
    mode = input("Quel mode choisir ? ")
    print(mode)
    if ( mode == 1 ) :
        query = input("Entrer la chaine de charactere a tracker : ")
        tracker(query)
    if mode == 2 :
        basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)  
        filename = input(" Nom fichier json : ")
        json2redis(filename,basejson)
    if mode == 3 :
        basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
        filename = input(" Nom fichier json : ")
        jsonfile = redis2json(filename,basejson)

        for line in jsonfile:            
            print('line')
            print(type(line))
            # tweet = json2tweet(line)
            text = getTweetText(line.decode('unicode-escape'))
            print(text)
            
    if mode == 4 :
        print('1 : tokeniser un fichier .txt')
        print('2 : tokeniser un une chaine de caractere')
        sousmode = input("Quel mode choisir ? ") 
        if sousmode == 1 :        
            filename = input("Quel est le nom du fichier ou chemain d'acces ? ")
            try :
                with open(filename,'r') as f:    
                    for line in f:        
                        tokens = text2tokens(line.decode('unicode-escape'),"s")
                        print(tokens)
            except :
                print(" Erreur 41 ")
                print("le nom du fichier doit être de la forme 'monfichier.txt' ou '/sousdossier/monfichier.txt' encodé en ANSII")

        if sousmode == 2 :        
            chaine = input("Entrez la chaine de caractere :  ")
            try :                                        
                tokens = text2tokens(chaine.decode('unicode-escape'),"s")
                print(tokens)
            except :
                print(" Erreur 42 ")
                pass


    if mode == 5 :
        filename = input("Quel est le nom du fichier ou chemin d'acces ? ")
        lda = txt2lda(filename)
    
    if mode == 6 :
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
    if mode == 7 :#Compter le nombre de tweet dans un json
        filename = input(" Entrer le ficher a compter  : ")
        count = countTweetInJson(filename)
        print(count)
        
