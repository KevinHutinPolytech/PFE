# coding: utf8
from __future__ import unicode_literals
from unidecode import unidecode
import nltk

text = "Bonjour, aujourd'hui semble être une belle journée"
textok = unidecode(text)
for sent in nltk.sent_tokenize(textok):
    print(list(nltk.pos_tag(nltk.word_tokenize(sent))))
    print()
    
import json
def read_file(file):
        for line in file:
            yield unidecode(json.loads(line)["text"])

from gensim import corpora
with open ('stream_StopMacron.json', 'r') as f:
        dictionary = corpora.Dictionary(line.strip().split() for line in read_file(f))


class MyCorpus(object):
    def __iter__(self):
        for line in read_file(open ('stream_StopMacron.json', 'r')):
            yield dictionary.doc2bow(line)

from gensim import models
lda = models.ldamodel.LdaModel(corpus = MyCorpus(), id2word = dictionary, num_topics = 100, update_every = 1, chunksize = 100000, passes = 3)
print(lda)
print(type(lda))
