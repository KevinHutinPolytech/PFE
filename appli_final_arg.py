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
    
    with open(monfichier,'r',encoding='utf-8',errors='replace') as f:    
        #texts = [[tokens for tokens in text2tokens(line.decode('unicode-escape'),"s") if len(tokens) != 0 ] for line in f ]
        texts = [[tokens for tokens in text2tokens(line, "s") if len(tokens) != 0] for line in f]
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
        doc_bow = dictionary.doc2bow(text2tokens(doc,"s"))
        print(lda[doc_bow]) # get topic probability distribution for a document
        vec_lda = lda[doc_bow]
        for i in vec_lda : 
            topicid = i[0]
            print(i[0])
            print(lda.get_topic_terms(topicid, topn=10))
        
        f.close()
    return lda
       
def updateDocAllwords(filename,topic,documents,allwords):
    texts = []
    frequency = defaultdict(int)
    try:
        with open(filename,'r',encoding='utf-8',errors='replace') as f:    
            for line in f :
                documents.append((line,topic))
                texts.append([tokens for tokens in text2tokens(line,"t") if len(tokens) != 0 ])

            # remove words that appear only once   
            for text in texts:
                for token in text:
                    frequency[token] += 1
            for text in texts:       
                for token in text :
                    if frequency[token] > 1:
                        allwords.append(token)
            save_documents = open("documents.pickle","wb")
            pickle.dump(documents, save_documents)
            save_documents.close()
    except: 
        print("Erreur dans updateDocAllwords \n","filename : ",filename ,"topic : ",topic,"documents : ",documents,"allwords : ",allwords)
        
def find_features(document):
    words = text2tokens(document,"t")
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features # Retourne un dict ou chaque mot est une clé    
       


######################################## MAIN ################################


mode = sys.argv[1]

if mode == '-h':
    print('Modes:')
    print("1 : Tracker des tweet sur twitter ")
    print("2 : Stocker JSON dans redis ")
    print("3 : Recuperer texte d'un JSON provenant de redis ")
    print("4 : tokeniser un text ")
    print('5 : Créer un corpus avec le model LDA')    
    print('6 : Track une chaine dans tweeter et donne une liste de dict{id_tweet : , tokens: , stems: , topic: }')
    print('7 : Compte le nombre de tweet dans un fichier json')
    print('8 : Classifier des documents')
    
   
if mode == 1 :
    query = eval(input("Entrer les mots cle a tracker : "))
    words = [word for word in query.split()]
    tracker(words)
if mode == 2 :
    basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)  
    filename = eval(input(" Nom fichier json : "))
    json2redis(filename,basejson)
if mode == 3 :
    basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
    filename = eval(input(" Nom fichier json : "))
    jsonfile = redis2json(filename,basejson)

    for line in jsonfile:            
        print('line')
        print(type(line))
        # tweet = json2tweet(line)
        text = getTweetText(line)#.decode('unicode-escape')
        print(text)

if mode == 4 :
    print('1 : tokeniser un fichier .txt')
    print('2 : tokeniser un une chaine de caractere')
    sousmode = eval(input("Quel mode choisir ? "))
    if sousmode == 1 :        
        filename = eval(input("Quel est le nom du fichier ou chemain d'acces ? "))
        try :
            with open(filename,'r') as f:    
                for line in f:        
                    tokens = text2tokens(line.decode('unicode-escape'),"s")
                    print(tokens)
        except :
            print(" Erreur 41 ")
            print("le nom du fichier doit être de la forme 'monfichier.txt' ou '/sousdossier/monfichier.txt' encodé en ANSII")

    if sousmode == 2 :        
        chaine = eval(input("Entrez la chaine de caractere :  "))
        try :                                        
            tokens = text2tokens(chaine.decode('unicode-escape'),"s")
            print(tokens)
        except :
            print(" Erreur 42 ")
            pass


if mode == 5 :
    filename = eval(input("Quel est le nom du fichier ou chemin d'acces ? "))
    lda = txt2lda(filename)

if mode == 6 :
    wordkey = eval(input("Entrer le nom du fichier json a ouvrir (sans le .json)"))
    topic = eval(input("Entrer le topic dans lequel s'inscrit ce mot cle : "))
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
    filename = eval(input(" Entrer le ficher a compter  : "))
    count = countTweetInJson(filename)
    print(count)
 if mode == 8 :#Classify
    print("Pour le moment seul 'Emploi.txt' et 'economie.txt' peuvent être exploite pour ce module")
    all_words = []
    documents = []
    addDoc = 1
   ''' while addDoc :            
        filename = eval(input("Entrer le nom du fichier a uploader : "))
        topic = eval(input("Entrer le topic du document : "))
        updateDocAllwords(filename,topic,documents,all_words)
        addDoc = eval(input("Voulez vous ajouter un document ? (1: oui | 0: non)"))
'''
    filename = sys.argv[2]
    topic = sys.argv[3]
    updateDocAllwords(filename,topic,documents,all_words)
    
    filename = sys.argv[4]
    topic = sys.argv[5]
    updateDocAllwords(filename,topic,documents,all_words)
    
    all_words_dict = nltk.FreqDist(all_words)
    #print("ALL_WORDS : ", all_words_dict)

    word_features = list(all_words_dict.keys())[:5000]
    #print("word_features : ", word_features)

    save_word_features = open("word_features5k.pickle","wb")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()

    featuresets = [(find_features(rev),categorie) for (rev,categorie) in documents]# Retourne une liste de dict ou chaque mot est une clé
    #print("featuresets : ", featuresets)

    random.shuffle(featuresets)
    #print(len(featuresets))

    testing_set = featuresets[10000:]
    training_set = featuresets[:10000]

    try :
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
        classifier.show_most_informative_features(15)

        ###############
        save_classifier = open("originalnaivebayes5k.pickle","wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()
    except :
        print("Pb dans le NaiveBayesClassifier")

    try :
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)
        print(LogisticRegression_classifier)
        LogisticRegression_classifier.fit(training_set)
        print(LogisticRegression_classifier)
        #print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
        LogisticRegression_classifier.show_most_informative_features(15)

        save_classifier = open("LogisticRegression_classifier5k.pickle","wb")
        pickle.dump(LogisticRegression_classifier, save_classifier)
        save_classifier.close()
    except :
        print("Pb dans le LogisticRegression_classifier")



