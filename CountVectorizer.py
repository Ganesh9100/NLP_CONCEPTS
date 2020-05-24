#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:18:05 2020

@author: ganesh
"""
import nltk

paragraph =  """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""
import nltk 
import re 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer

ps=PorterStemmer()
lem=WordNetLemmatizer()

# changing the para to sentence 
sentence=nltk.sent_tokenize(paragraph)
corpus=[]
# first sentence is filtered with only ato z and A to Z 
# then lower the sentence 
# then split the sentence which is then stored as list as default . so the list contains the words in cucrrent sentence 
# then using for loop for iteratng the word in one sentence , 
# if word is not in stopwords then lemmatise it and store it in reiew 
# then the review of all words is joined and  again putting as sentence and at last appended to corpus 
    
for i in range(len(sentence)):
    review=re.sub('[^a-zA-Z]',' ',sentence[i])
    review=review.lower()
    review=review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   
            
    review =' '.join(review) 
    corpus.append(review)
            

#this is used in small data set only , 
# in bag of words , we dont have sematic meaning ie after reprecentation of matrix , it will have either 0 or 1 or more based on repeation , but this wont help in giving meaning 
# but here , this tfidfvectorizer will make use of the formula (tf x idf ), that will give sematic meaning liek 
# eg , good boy girl here it will have weightage for boy and girl than good , and boy adn girl value will nearly be same 
# WORD2VEC IS BETTER THAN TFIDF
from sklearn.feature_extraction.text import CountVectorizer   
tf=CountVectorizer()
x=tf.fit_transform(corpus).toarray()
    