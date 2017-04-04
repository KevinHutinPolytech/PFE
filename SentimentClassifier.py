
if sys.argv[2] == '-h' or sys.argv[2] == '':
            print("monfichier1.txt topic1 monfichier2.txt topic2 : Nom des fichier .txt a traiter suivie de leurs topics")                   
            print("Pour le moment seul 'Emploi.txt' et 'economie.txt' peuvent être exploite pour ce module")
        else : 
            all_words = []
            documents = []
            addDoc = 1
            filename = sys.argv[2]
            topic = sys.argv[3]
            updateDocAllwords(filename,topic,documents,all_words)

            filename = sys.argv[4]
            topic = sys.argv[5]
            updateDocAllwords(filename,topic,documents,all_words)

            all_words_dict = nltk.FreqDist(all_words)
            #print("ALL_WORDS : ", all_words_dict)
            '''
            word_features = list(all_words_dict.keys())[:500]
            #print("word_features : ", word_features)
            
            save_word_features = open("word_features500.pickle","wb")
            pickle.dump(word_features, save_word_features)
            save_word_features.close()
            '''
            word_features_lda = []
            ####################### Document 1 ##########################
            lda_model = txt2lda(sys.argv[2])            
            #retourne list de tuple (idtopic, [liste2]) où [liste2] est une liste de tuple (word, probability)
            lda_features = lda_model.show_topics(num_topics=20, num_words=15, log=False, formatted=False)
            #print("lda_features : ",lda_features)
            
            for topic in lda_features :
                #print("Topic ",topic[0],": ", topic) 
                for word in topic[1] :
                    #print("Word: ", word)
                    word_features_lda.append(word[0])
            
            
            ####################### Document 2 ##########################
            lda_model_2 = txt2lda(sys.argv[4])            
            #retourne list de tuple (idtopic, [liste2]) où [liste2] est une liste de tuple (word, probability)
            lda_features_2 = lda_model_2.show_topics(num_topics=20, num_words=15, log=False, formatted=False)
            #print("lda_features : ",lda_features_2)            
            for topic in lda_features_2 :
                #print("Topic ",topic[0],": ", topic) 
                for word in topic[1] :
                    #print("Word: ", word)
                    word_features_lda.append(word[0])
                    
            print("Word Features lda (Size :",len(word_features_lda),"): ", word_features_lda)
            save_word_features = open("word_features_lda.pickle","wb")
            pickle.dump(word_features_lda, save_word_features)
            save_word_features.close()
            
            featuresets = [(find_features(rev,word_features_lda),categorie) for (rev,categorie) in documents]# Retourne une liste de dict ou chaque mot est une clé
            #print("featuresets : ", featuresets)

            random.shuffle(featuresets)
            #print(len(featuresets))

            testing_set = featuresets[300:]
            training_set = featuresets[:300]

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

            LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
            LogisticRegression_classifier.train(training_set)
            print("sklearn classifier créer en LogisticRegression : \n",LogisticRegression_classifier)
            #LogisticRegression_classifier.fit(training_set)
            #print(LogisticRegression_classifier)
            print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

            print("Labels :",LogisticRegression_classifier.labels())
            print ("Type : ",type(LogisticRegression_classifier))
            
            dictum = [tupl[0] for tupl in testing_set]
            print("dictum : ",dictum)
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
    
            
            save_classifier = open("LogisticRegression_classifier.pickle","wb")
            pickle.dump(LogisticRegression_classifier, save_classifier)
            save_classifier.close()
            '''
            except Exception as e  :
                print("Pb dans le LogisticRegression_classifier")
                print(Exception)
            '''

