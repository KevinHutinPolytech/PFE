##################################BIBLIOTEQUE################################
# coding: utf8
from __future__ import unicode_literals
import operator 
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


#############################FONCTIONS####################################

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
def txt2lda(monfichier):    
    
    with open(monfichier,'r',encoding='utf-8',errors='replace') as f:    
        #texts = [[tokens for tokens in text2tokens(line.decode('unicode-escape'),"s") if len(tokens) != 0 ] for line in f ]
        texts = [[tokens for tokens in text2tokens(line, "t") if len(tokens) != 0] for line in f]
        # remove words that appear only once            
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        dictionary = corpora.Dictionary(texts)
        dictionary.save('/tmp/emploi.dict')  # store the dictionary, for future reference
        
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize('/tmp/emploi.mm', corpus)  # store to disk, for later use
        
        
        lda = models.LdaModel(corpus, id2word=dictionary, num_topics=20)
        print("Génération d'un model LDA...")
        pprint(lda)
        '''
        print(lda.print_topics(num_topics=20,num_words=75))
        print(lda.get_topic_terms(19, topn=10))
        print(type(lda.get_topic_terms(19, topn=10)))
        '''
        print("LDA généré")
        
        # Retourne une liste de tuple (idtopic , "0.02*mot1 + 0.02*mot2 + ... + 0.00*motn")
        #print("Show_Topics :lda.show_topics(num_topics=10, num_words=10, log=False, formatted=True)",lda.show_topics(num_topics=10, num_words=10, log=False, formatted=True))
        
        #retourne list de tuple (idtopic, [liste2]) où [liste2] est une liste de tuple (word, probability)      
        #print("Show_Topics :lda.show_topics(num_topics=10, num_words=10, log=False, formatted=False)",lda.show_topics(num_topics=10, num_words=10, log=False, formatted=False))
        '''
        doc = " Le marche de l'emploi est en chute libre, le nombre de chomeur ne cesse d'augmenter "
        doc_bow = dictionary.doc2bow(text2tokens(doc,"s"))
        print(lda[doc_bow]) # get topic probability distribution for a document
        vec_lda = lda[doc_bow]
        for i in vec_lda : 
            topicid = i[0]
            print(i[0])
            print(lda.get_topic_terms(topicid, topn=10))
        '''
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
        
def find_features(document,word_features):
    words = text2tokens(document,"t")
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features # Retourne un dict ou chaque mot est une clé    


######################################## MAIN ################################


mode = sys.argv[1]
            
if mode == '-h':
    print('Passez en argument pos.txt neg.txt')
else :   
    all_words = []
    documents = []            
    filename = sys.argv[1]
    topic ="positif"
    updateDocAllwords(filename,topic,documents,all_words)

    filename = sys.argv[2]
    topic = "negatif"
    updateDocAllwords(filename,topic,documents,all_words)

    all_words_dict = nltk.FreqDist(all_words)          

    word_features_lda = []
    ####################### Document 1 ##########################
    #Génère un model Bag of word LDA positif
    lda_model = txt2lda(sys.argv[1])            
    #retourne list de tuple (idtopic, [liste2]) où [liste2] est une liste de tuple (word, probability)
    lda_features = lda_model.show_topics(num_topics=20, num_words=15, log=False, formatted=False)
    #print("lda_features : ",lda_features)

    for topic in lda_features :
        #print("Topic ",topic[0],": ", topic) 
        for word in topic[1] :
            #print("Word: ", word)
            word_features_lda.append(word[0])


    ####################### Document 2 ##########################
    #Génère un model Bag of word LDA négatif
    lda_model_2 = txt2lda(sys.argv[2])            
    #retourne list de tuple (idtopic, [liste2]) où [liste2] est une liste de tuple (word, probability)
    lda_features_2 = lda_model_2.show_topics(num_topics=20, num_words=15, log=False, formatted=False)
    #print("lda_features : ",lda_features_2)            
    for topic in lda_features_2 :
        #print("Topic ",topic[0],": ", topic) 
        for word in topic[1] :
            #print("Word: ", word)
            word_features_lda.append(word[0])

    #print("Word Features lda (Size :",len(word_features_lda),"): ", word_features_lda)
    save_word_features = open("word_features_sentiment_lda.pickle","wb")
    pickle.dump(word_features_lda, save_word_features)
    save_word_features.close()

    featuresets = [(find_features(rev,word_features_lda),categorie) for (rev,categorie) in documents]# Retourne une liste de dict ou chaque mot est une clé
    #print("featuresets : ", featuresets)

    random.shuffle(featuresets)
    print("nombre de mot pertinant : ",len(featuresets))

    testing_set = featuresets[len(featuresets)/2:]
    training_set = featuresets[:len(featuresets)/2]

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("sklearn classifier créer en LogisticRegression : \n",LogisticRegression_classifier)
    #LogisticRegression_classifier.fit(training_set)
    #print(LogisticRegression_classifier)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    print("Labels :",LogisticRegression_classifier.labels())           

    dictum = [tupl[0] for tupl in testing_set]            
    try :
        print("classify many:" , LogisticRegression_classifier.classify_many(dictum)) 
    except :
        print("classify many erreur \n","Type testing_set: ",type(dictum),"\n testing_set :",dictum) 

    try :
        print("prob_classify_many:" , LogisticRegression_classifier.prob_classify_many(dictum))
        for probdisti in LogisticRegression_classifier.prob_classify_many(dictum):
            list_of_samples = probdisti.samples()
            for sample in list_of_samples:
                print("Sample: ", sample, " Prob : ",probdisti.prob(sample))
    except :
        print("prob_classify_many erreur \n","Type testing_set:",type(dictum)) 


    save_classifier = open("Sentiment_classifier.pickle","wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()


