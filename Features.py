import torch.nn as nn
from utils.masked_loss import masked_cross_entropy
from utils.statistics import JointLossStatistics
from utils.time_log import time_since
from validation import evaluate_loss
import time
from nltk.stem.porter import *
import math
import logging
import torch
import sys
import os
import pickle
import metric
import numpy as np
import math
import nltk
import operator
from nltk.corpus import stopwords
from rake_nltk import Rake
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
def Feature1(ReviewText):
	result=np.array([])
	S=len(ReviewText)
	if S==1:
                result=np.append(result,1)
                return result 
	if(S%2==0):
		d=float(S)
		d=(d+1)/2
	else:
		d=float(S)
		d=(d+1)/2
	for i in range(0,S):
        	result=np.append(result,abs(d - (i+1)))
        
	norm=np.linalg.norm(result)

	result=result/(d-1)
#	print(result)
#	print(ReviewText)
	return result

def Feature2(ReviewText):
	result=np.array([])
	for sent in ReviewText:
		tokens = nltk.word_tokenize(sent.lower())
		text = nltk.Text(tokens)
		tags = nltk.pos_tag(text)
		#print(tags)
		counts = Counter(tag for word,tag in tags)
		total = sum(counts.values())
		ct=counts['JJ']
		ct=float(ct)
		#print(ct)
		total=float(total)
		result=np.append(result, ct/total)
#	print(result)
	return result

def Feature3(ReviewText):
	result=np.array([])
	for sent in ReviewText:
		if "but" in sent:
			result=np.append(result,1)
		else:
			result=np.append(result,0)
#	print("&&&&&&&&&&&&&&&&&&&&")
#	print(result)
	return result

def Feature4(ReviewText):
        result=np.array([])
        c=0.0
        for sent in ReviewText:
                c=0.0
                tokens=sent.split()
                tw=len(tokens)
                tw=float(tw)
                for x in tokens:
                                if(x.isnumeric()):
                                        c+=1
                #print(sent)
                #print(c)
                #print(tw)
                result=np.append(result,c/tw)
        maxv=np.amax(result)
        if maxv>0.0:
        	result=result/maxv
        return result
#
 #                                                                          l1=[];l2=[]                                                                                                                                                                                                                                  x_set={w for w in x if not w in sw}                                                                                                                                                                                                          y_set={w for w in y if not w in sw}                                                                                                                                                                                                          rvector=x_set.union(y_set)                                                                                                                                                                                                                   for w in rvector:                                                                                                                                                                                                                                    if w in x: l1.append(1)                                                                                                                                                                                                                      else: l1.append(0)                                                                                                                                                                                                                           if w in y: l2.append(1)                                                                                                                                                                                                                      else: l2.append(0)                                                                                                                                                                                                              
def Feature5(ReviewText):
        result=np.array([])
        for sent1 in ReviewText:
                ct=0.0
                for sent2 in ReviewText:
                        if sent1 !=sent2:
                                if CosineUtil(sent1,sent2)>0.20:
                                        ct+=1
                result=np.append(result,ct)
        maxv=np.amax(result)
        if maxv>0.0:
        	result=result/maxv
        return result

 
# def Feature4(ReviewText):
# 	for sent in ReviewText:

def CosineUtil(s1, s2):        
	x= nltk.word_tokenize(s1)
	y=nltk.word_tokenize(s2)
	sw=stopwords.words('english')
	l1=[];l2=[]
	x_set={w for w in x if not w in sw}
	y_set={w for w in y if not w in sw}
	rvector=x_set.union(y_set)
	for w in rvector:
                if w in x: l1.append(1)
                else: l1.append(0)
                if w in y: l2.append(1)
                else: l2.append(0)
	c=0
	for i in range(len(rvector)):
                c+= l1[i]*l2[i]
	cosine=c/float((sum(l1)*sum(l2))**0.5)
	return cosine# 		for x in corpus:
# 			if x in sent:

# def Feature5(ReviewText):
def Feature6(ReviewText):
        result=np.array([])
        score=0.0
        for sent1 in ReviewText:
                score=0.0
                for sent2 in ReviewText:
                        if sent1 != sent2:
                                score+=CosineUtil(sent1,sent2)
                result=np.append(result,score)
        #print("RESULT")
        #print(result)
        maxv=np.amax(result)
        if maxv>0.0:
        	result=result/maxv
        return result

def Feature7(ReviewText):
        result=np.array([])
        fulltext=""
        counts=dict()
        for sent in ReviewText:
                fulltext=fulltext+" "+sent
        sw=stopwords.words('english')
        words=nltk.word_tokenize(fulltext)
        punctuations = '`~!@#$%^&*()_+{}|:"<>?-=[]``\;\'.\/,'
        for word in words:
                if word.isalpha():
                        word=word.lower()
                if word in counts.keys() and word not in punctuations:

                    counts[word] += 1
                else:
                    counts[word]=1
        counts2=dict()
        for keys in counts:
                if keys not in sw:
                        counts2[keys]=counts[keys]
        sorted_d = dict( sorted(counts2.items(), key=operator.itemgetter(1),reverse=True))
        c=0
        l=[]
        #print("COUNTS")
        #print(counts)
        #print("COUNTS@")
        #print(sorted_d)
        for key in sorted_d:
                if c>5:
                        break
                l.append(key)
                c+=1
        #print(l)
        for sent in ReviewText:
                ct=0.0
                
                words=nltk.word_tokenize(sent)
                tw=len(words)
                for word in words:
                        if word.isalpha():
                                word=word.lower()
                        if word in l:
                                ct+=1
                result=np.append(result, ct/tw)
        maxv=np.amax(result)
        result=result/maxv
        return result

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = nltk.word_tokenize(sent)
        for word in words:
            if word.isalpha():
            	word = word.lower()
            	word = ps.stem(word)
            if word in stopWords:
               continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent] = freq_table

    return frequency_matrix
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = []

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue.append(total_score_per_sentence / count_words_in_sentence)

    return sentenceValue
def Feature8(ReviewText):
        result=np.array([])
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(ReviewText)
        X=X.toarray()
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
        #print(X)
        for sent in X:
                result=np.append(result, np.sum(sent))
        maxv=np.amax(result)
        result=result/maxv
        return result

def Feature9(ReviewText):
        result=np.array([])
        with open('pos.pkl','rb') as f:
                pos=pickle.load(f)
        #for w in pos:
         #       print(type(w))
        #print("CHECKING IF")
        #print(type(pos))
        #if "love" in pos:
        #        print("YESSSSSSSSSSSSSSSSSSSSS")
        c=0.0
        for sent in ReviewText:
                sent2=sent.lower()
		
                words=nltk.word_tokenize(sent2)
                tw=len(words)
                for w in words:
                	if  w in pos:
                		c+=1
                result=np.append(result, c/tw)
        maxv=np.amax(result)
        if maxv>0:
        	result=result/maxv
        return result

def Feature10(ReviewText):
	result=np.array([])
	with open('neg.pkl', 'rb') as f:
		neg=pickle.load(f)
	c=0.0
	for sent in ReviewText:
		sent2=sent.lower()
		words=nltk.word_tokenize(sent2)
		tw=len(words)
		for w in words:
			if w in neg:
				c+=1
		result=np.append(result,c/tw)
	maxv=np.amax(result)
	if maxv>0:
		result=result/maxv
	return result
#print(count_doc_per_words
