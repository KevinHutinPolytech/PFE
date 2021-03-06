from gensim import corpora
import os
documents = ["La justice pour tous demande une jurice digne de ce nom ",
              "Pauvre theo qui ne demande que justice soit faite",
              "PSG du lourd ce soir, au chiotte marseille",
             "Si il ne savent pas jouer avec un ballon autant arreter le foot",
             "une honte pour lhistoire du football",
             "La planete se meure et personne ne sen preocupe",
             "Foret devastee, crime contre l humanite ou va lavenir de cette planete",
             "Nul doute il sest fais tuer",
              "disparition de Tom Viguer, le respect a disparue"]

# remove common words and tokenize
stoplist = set('le la pour de a les et dans'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

from pprint import pprint  # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)

print(dictionary.token2id)

new_doc = "Quel sentence appliquer au criminel, qui se battent contre la justice"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
print(corpus)


#######################TUTO2###########################################

print('tuto2')
from gensim import corpora, models, similarities
if (os.path.exists("/tmp/deerwester.dict")):
   dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
   corpus = corpora.MmCorpus('/tmp/deerwester.mm')
   print("Used files generated from first tutorial")
else:
   print("Please run first tutorial to generate data set")


model = models.LdaModel(corpus, id2word=dictionary, num_topics=2)

pprint(model)

#######################TUTO3###########################################
print('tuto3')
from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
corpus = corpora.MmCorpus('/tmp/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

print("")
print("LDA")
print("")

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2)

doc = "Quel sentence appliquer au criminel, qui se batte contre la justice"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lda = lda[vec_bow] # convert the query to LSI space
print(vec_lda)

index = similarities.MatrixSimilarity(lda[corpus]) # transform corpus to LSI space and index it

index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

sims = index[vec_lda]# perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples

print("")
print("LSI")
print("")

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = "Quel sentence appliquer au criminel, qui se batte contre la justice"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

sims = index[vec_lsi]# perform a similarity query against the corpus
print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples
