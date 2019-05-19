import pandas as pd
import numpy as np
import re
import os
import nltk
import nltk.data
import logging

from bs4 import BeautifulSoup  
from nltk.corpus import stopwords 
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


def review_to_wordlist( raw_review, remove_stopwords=False):
    review_text = BeautifulSoup(raw_review, "lxml").get_text()  # 1. Remove HTML    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)        # 2. Remove non-letters 
    words = letters_only.lower().split()                        # 3. Convert to lower case, split into individual words                            
    if remove_stopwords:                                        # 4. Convert the stop words to a set   
        stops = stopwords.words("english")                      # 4. Convert the stop words to a set 
        stops.extend(['movie','film'])    
        stop_word = set(stops) 
        words = [w for w in words if not w in stop_word]        # 5. Remove stop words
    return( " ".join( words ))                                  # 6. Join the words back into one string separated by space 

def review_to_sentences( review, tokenizer, remove_stopwords = False ):
    raw_sentences = tokenizer.tokenize(review.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    index2word_set = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    
    for review in reviews:
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)

       counter = counter + 1.
    return reviewFeatureVecs



if __name__ == '__main__':

	data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

	from sklearn.model_selection import train_test_split
	train, test = train_test_split(data, test_size=0.25)

	train.index = np.arange(len(train))
	test.index = np.arange(len(test))
	
	print ("Read %d labeled train reviews, %d labeled test reviews\n" % (train["review"].size, test["review"].size))

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = []

	print ("Parsing sentences from training set")
	for review in train["review"]:
		sentences += review_to_sentences(review, tokenizer)

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


	print ("Training Word2Vec model...")

	model = Word2Vec(sentences, workers=num_workers, size=200)
	model.init_sims(replace=True)

	model_name = "200features"
	model.save(model_name)



	print ("Creating average feature vecs for training reviews")
	clean_train_reviews = []
	for review in train["review"]:
		clean_train_reviews.append( review_to_wordlist( review))
	
	trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features)

	print ("Creating average feature vecs for test reviews")
	clean_test_reviews = []
	for review in test["review"]:
		clean_test_reviews.append( review_to_wordlist( review))

	testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features)

	logreg = LogisticRegression(C = 100, random_state = 0)


	print ("Fitting a logreg to training data...")

	logreg = logreg.fit( trainDataVecs, train["sentiment"])

	result = logreg.predict_proba( testDataVecs )[:,1]
	predict = logreg.predict_proba( trainDataVecs )[:,1]

	#output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	#output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )


	fpr, tpr, threshold = metrics.roc_curve(test["sentiment"], result)
	roc_auc = metrics.auc(fpr, tpr)

	plt.title('Word2Vec')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('word2vec_roc_test.png')
	plt.show()

	fpr1, tpr1, threshold1 = metrics.roc_curve(train["sentiment"], predict)
	roc_auc1 = metrics.auc(fpr1, tpr1)

	plt.title('Word2Vec')
	plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig('word2vec_roc_train.png')
	plt.show()

print ("Wrote Word2Vec_AverageVectors.csv")