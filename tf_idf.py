import pandas as pd
import numpy as np
import sklearn
import re
import nltk

from bs4 import BeautifulSoup  
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import roc_auc_score

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review, "lxml").get_text()  # Удаляем HTML    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)        # Удаляем не буквенные символы
    words = letters_only.lower().split()                        # Преобразуем в нижний регистр и разделим на отдельные слова                           
    stops = stopwords.words("english")                          
    stops.extend(['movie','film'])                              # добавим 'movie' и'film' в стоп-слова 
    stop_word = set(stops)                                      # Конвертируем стоп-слова в множество
    meaningful_words = [w for w in words if not w in stops]     # Remove stop words
    return( " ".join( meaningful_words))                        # Join the words back into one string separated by space 



data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)

train.index = np.arange(len(train))
test.index = np.arange(len(test))

print ("Cleaning and parsing the training set movie reviews...\n")

num_reviews = train['review'].size
clean_train_reviews = []
for i in range( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews))                                                                    
    clean_train_reviews.append(review_to_words(train['review'][i]))



print ("Creating the bag of words...\n")

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 200) 
train_data_features = vectorizer.fit_transform(clean_train_reviews)

tfidf = TfidfTransformer()
tfidf_features = tfidf.fit_transform(train_data_features)

vocab = vectorizer.get_feature_names()

dist = np.sum(tfidf_features, axis=0)
for tag, count in zip(vocab, dist):
    print (count, tag)



print ("Training the Logistic Regression...\n")

logreg = LogisticRegression(C = 100, random_state = 0)
logreg = logreg.fit(tfidf_features, train['sentiment'])


print ("Cleaning and parsing the test set movie reviews...\n")

num_reviews = len(test['review'])
clean_test_reviews = [] 
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test['review'][i] )
    clean_test_reviews.append( clean_review )

test_data_features = vectorizer.transform(clean_test_reviews)
test_tfidf_features = tfidf.fit_transform(test_data_features)
result = logreg.predict_proba(test_tfidf_features)[:,1]

#output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
#output.to_csv( "Bag_of_Words_model_tfidf.csv", index=False, quoting=3 )

predict = logreg.predict_proba(tfidf_features)[:,1]
fpr, tpr, threshold = metrics.roc_curve(train['sentiment'], predict)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('TF-IDF')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('tfidf_roc_train.png')
plt.show()

fpr1, tpr1, threshold1 = metrics.roc_curve(test['sentiment'], result)
roc_auc1 = metrics.auc(fpr1, tpr1)

plt.title('TF-IDF')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('tfidf_roc_test.png')
plt.show()


