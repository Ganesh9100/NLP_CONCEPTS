#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:59:55 2020

@author: ganesh
"""

import pandas as pd

mail = pd.read_csv('/home/ganesh/Desktop/SpamClassifier-master/smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
# the above note pad is 2 parts 1st column represent the lable spam or ham 
# then the dependent var and independent var is is sepereated by one tab so /t

#and there is no column name so im forcingly specifying 2 heading . 1st is lable and 2nd is message

# now data cleaning and pre processssssssssngs

import nltk
import re
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')
from nltk.corpus import stopwords 

ps=PorterStemmer()# stemming purpose 
lem=WordNetLemmatizer()
corpus=[]

for i in range(0,len(mail)):
    
    review = re.sub('[^a-zA-z]',' ',mail['message'][i])# space is given in 2nd parameter of sub 
    review=review.lower()
    review=review.split()
    
    review=[lem.lemmatize(word) for word in review if not word in stopwords.words('english')]
    
    review=' '.join(review) # words int o sentence 
    corpus.append(review)
    
    
#when i use TfidfVectorizer , i get accuracy of 0.9778708133971292
#from sklearn.feature_extraction.text import TfidfVectorizer
#cv=TfidfVectorizer(max_features=5000)# here the max_feature will take top 5k most important word  up to 5000 words 
#x=cv.fit_transform(corpus).toarray()# this is the date that we will be training 
#print("=============")

#when i use CountVectorizer bag of words  , i get accuracy of 0.9850478468899522
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)# here the max_feature will take top 5k most important word  up to 5000 words 
x=cv.fit_transform(corpus).toarray()# this is the date that we will be training 
print("=============")
# we have target var as ham and spam so we gonna make it mumentrical as 1 for spam and 0 for ham 

y=pd.get_dummies(mail['label'])

# here this y cnsisit of 1 as ham and 0 as spam , and it has 2 columns differently so we re going to make it one column 
y=y.iloc[:,1].values  # now 0 means its ham and 1 means it spam 

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


#=================================================================
# using naves bayes 0.9850478468899522

#from sklearn.naive_bayes import MultinomialNB
#spam_detect=MultinomialNB()
#train=spam_detect.fit(x_train,y_train)
#y_pred=spam_detect.predict(x_test)

#=================================================================
# using nLogisticRegression  i get accuracy 0.9820574162679426
from sklearn.linear_model import LogisticRegression
lrg=LogisticRegression()
lrg.fit(x_train,y_train)
y_pred=lrg.predict(x_test)


#=================================================================
# using DecisionTreeClassifier  i get accuracy 0.9360047846889952
#from sklearn.tree import DecisionTreeClassifier
#clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
#clf_entropy.fit(x_train,y_train)
#y_pred=clf_entropy.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)

print("Using CountVectorizer and with Naive Bayes model",acc)
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
import seaborn as sns 

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,fmt=".2f",linewidths=.10,square=True,cmap='Blues_r')
plt.ylabel("ACTUAL VALUE OR LABEL")
plt.xlabel(" PREDICTED VALUE OR LABEL")
all_sample_title="ACCURACY SCORE {0}".format(acc)
plt.title(all_sample_title,size=15)

