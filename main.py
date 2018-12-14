#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 15:08:53 2018

@author: huwei

Project 3 Natural Language Processing with Women’s Clothing E-Commerce dataset

Data source: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

"""

# General
import pandas as pd
import numpy as np
import time
import os

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

# Preprocessing
import string
from nltk.tokenize import RegexpTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
# Regualr expression operator
import re
#from nltk.stem import PorterStemmer

# Modeling
import statsmodels.api as sm
# Give a sentiment intensity score to sentences
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.util import ngrams
from collections import Counter
from gensim.models import word2vec
from sklearn import linear_model, metrics, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Warning
import warnings
warnings.filterwarnings('ignore')

from gensim import corpora, models, similarities, matutils
from gensim.corpora import Dictionary
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

from nltk.stem.lancaster import LancasterStemmer
# Vs. nltk.stem.PorterStemmer
from nltk.stem.porter import PorterStemmer

##############################################################################
# Self-defined functions
##############################################################################
# Define accuracy calculation function
def prediction_error(y, y_hat):
    """
    Calculate prediction accuaracy.
    
    Arguments:
        y  -- np.array / pd.Series
            The original value of the binary dependent variable;
        
        y_hat -- np.array / pd.Series
            The predicted value of the binary dependent variable;
            
    Return:
        error -- float
    """
    error = float(sum(y!=y_hat))/len(y)
    return error

# Generate cross validation error
def logitCV(X, y, Kfolds = 10):
    '''
    Calculate the K-fold cross validation error
    
    Arguments:
        X -- np.array
            The explanatory variables;
        y -- np.array
            The binary dependent variable;
        Kfolds -- integer
            The subsets number in the cross validation, default is 10;
            
    Return:
        cv_error -- float
            Cross validation prediction error;
    '''
    # initialized the container for the out-of-sample prediction  errors
    errors = np.zeros((Kfolds,), dtype = 'float32')
    # split the training data into 'Kfolds' subsets
    obs = len(y)
    training_split = [subset[1] for subset in \
                      cross_validation.KFold(obs, Kfolds, shuffle=True, random_state=1)]
    # Combine n-1 folds
    for i in range(Kfolds):
        test_idx = training_split[i]
        #pdb.set_trace()
        training_idx = [j for j in range(obs) if j not in test_idx]
        # logistic regression
        clf = linear_model.LogisticRegression()
        # fit the model
        clf.fit(X[training_idx], y[training_idx])
        # make prediction
        y_hat = clf.predict(X[test_idx,])
        errors[i] = prediction_error(y[test_idx], y_hat)
        
    cv_error = np.mean(errors)
    
    return cv_error

# Define the cross validation function for LDA
def ldaCV(texts, sentiment, n_topics):
    """
    Define the cross validation function for LDA
    
    Arguments:
        texts -- list
            The preprocessed texts;
        sentiment -- pandas.core.series.Series
            The sentiment of the text;
        n_topics -- numpy.ndarray
            The array of the number of topics for LDA;
            
    Return:
        n_topics[idx] -- numpy.int64
            Number of topics for LDA with lowest cross-validation error;
        best_lda -- gensim.models.ldamodel.LdaModel
            The LDA model with lowest cross-validation error;
        cv_errors -- numpy.ndarray
            The array of cross-validation error using different number of topics;
    """
    # Create dictionary
    dictionary = Dictionary(texts) 
    # convert text to BoW format corpus
    corpus = [dct.doc2bow(text) for text in texts]  
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    cv_errors = np.zeros((len(n_topics),), dtype = 'float32')
    lda = []
    for i in range(len(n_topics)):
        lda.append(models.LdaModel(corpus_tfidf, id2word = dictionary, num_topics = n_topics[i]))
        corpus_lda = lda[i][corpus_tfidf]
        matrix_lda = np.transpose(matutils.corpus2dense(corpus_lda, num_terms = n_topics[-1]))
        cv_errors[i] = logitCV(matrix_lda, sentiment)
        print('Number of topics:', n_topics[i], '; Cross validation error:', cv_errors[i])
        
    # Find the index for the smallest cv_errors
    val,idx = min((val,idx) for (idx, val) in enumerate(cv_errors))
    # Return the number of optimal corpus and the corresponding lda model 
    return n_topics[idx], lda[idx], cv_errors

#
def preprocessing(data):
    """
    Preprocess the textual data
    
    Arguments:
    data -- pandas.core.series.Series
        The texts data which need to be preprocessed;
        
    Return:
    words -- list
        list of all the preprocessed words in data;
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words if not w in stop_words] #3
    words = [st.stem(w) for w in words] #4
    return words

def get_ngrams(text, n):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    
    For example:
    >>> from nltk.util import ngrams
    >>> list(ngrams([1,2,3,4,5], 3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    
    Argument:
    texts -- pandas.core.series.Series
        The textual data;
    n -- int
        The number of ngrams;
    
    Return -- list
        list of all the ngrams of the data;
    """
    n_grams = ngrams((text), n)
    return [ ' '.join(grams) for grams in n_grams]

def gramfreq(text,n,num):
    """
    Get the frequency of ngrams in text data;
    
    Argument:
    text --
    n --
    num --
    
    Return:
    
    """
    # Extracting bigrams
    result = get_ngrams(text,n)
    # Counting bigrams
    result_count = Counter(result)
    # Converting to the result to a data frame
    df = pd.DataFrame.from_dict(result_count, orient='index')
    df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index column name
    return df.sort_values(["frequency"],ascending=[0])[:num]

def gram_table(data, gram, length):
    out = pd.DataFrame(index=None)
    for i in gram:
        table = pd.DataFrame(gramfreq(preprocessing(data),i,length).reset_index())
        table.columns = ["{}-Gram".format(i),"Occurrence"]
        out = pd.concat([out, table], axis=1)
    return out
##############################################################################
path_save_fig = os.path.join(os.getcwd(), 'Images')
path_save_data = os.path.join(os.getcwd(), 'Data')

##############################################################################
#### Read Data
##############################################################################
df = pd.read_csv(os.path.join(path_save_data,'Womens Clothing E-Commerce Reviews.csv')).fillna(' ')
print(df.columns)
print(df.index)
print(df.shape)

##############################################################################
#### Data Preprocessing
##############################################################################
pdtextpreprocess = df[['Title', 'Review Text', 'Rating']]

pdtextpreprocess['index'] = pdtextpreprocess.index

documents = pdtextpreprocess['Review Text']

# print
print(documents.shape)
word_tokenize(documents[0])
PorterStemmer().stem(word_tokenize(documents[0])[0])
type(stopwords)
#
#####################
# Text tokenized -- Unigram
print('Text Tokenization...')
t1 = time.time()
text_tokenized = [[word.lower() for word in word_tokenize(document)] \
                   for document in documents]
print('Time used: %s seconds' % (time.time()-t1))
#####################
# move stopwords
print('Removing Stopwords...')
t1 = time.time()
text_filtered_stopwords = [[word for word in document if not word in stopwords.words('english')]\
                           for document in text_tokenized]
print('Time used: %s seconds' % (time.time()-t1))
print('Length: %s' % len(text_filtered_stopwords))

#####################
# Removing English Punctuation
print('Removing English Punctuation...')
t1 = time.time()
english_punctuations = string.punctuation
texts_filtered = [[word for word in document if not word in english_punctuations] for document in text_filtered_stopwords]
print('Time used: %s seconds' % (time.time()-t1))
print('Length %s' % len(texts_filtered))

#####################
# Stemming
print('Stemming...')
st = PorterStemmer()
t1 = time.time()
texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]
print('Time used: %s seconds' % (time.time()-t1))
print('Length %s' % len(texts_stemmed))

print(texts_filtered[0])
print(texts_stemmed[0])

# list of all stems
all_stems = sum(texts_stemmed, [])
type(all_stems)
len(all_stems)

# count() method counts how many times an element has occurred in a list and returns it.
# set of all stems that appears only once
once_stems = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
type(once_stems)
len(once_stems)

#####################
#Removing the low frequence word (such as occurence equals one)
texts = [[stem for stem in text if stem not in once_stems] for text in texts_stemmed]
type(texts)
len(texts)

#####################
# save
with open(os.path.join(path_save_data,'Preprocessed Review Text.txt'), 'w') as f:
    for text in texts:
        for item in text:
            f.write("%s " % item)
            
        f.write("\n")
# Read

# =============================================================================
# texts = []
# with open(os.path.join(path_save_data,'Preprocessed Review Text.txt'), 'r') as f:
#     for line in f:
#         text = line.split()
#         texts.append(text)
# =============================================================================

# =============================================================================
# # Or you can use pickle
# import pickle
# 
# with open('Preprocessed Review Text.p', 'wb') as fp:
#     pickle.dump(texts, fp)    
# 
# with open ('Preprocessed Review Text.p', 'rb') as fp:
#     itemlist = pickle.load(fp)
# =============================================================================
##############################################################################
# Extracting topics and calculating texts correlation
##############################################################################
# Create dictionary
dct = Dictionary(texts) 
# convert text to BoW format corpus
corpus = [dct.doc2bow(text) for text in texts] 
# 
for cp in corpus[:1]:
    for id, freq in cp:
        print(dct[id],',', freq)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
type(corpus_tfidf)
for cp in corpus_tfidf[:1]:
    for id, freq in cp:
        print(dct[id],',', freq)
# =============================================================================
# for doc in corpus_tfidf:
#     print(doc,"\n")
# =============================================================================
# Latent Semantic Indexing
# Training lsi model
lsi = models.LsiModel(corpus_tfidf, id2word = dct, num_topics = 100)
lsi.print_topics(10)

# Map the document to the topic space to see the correlation between the document and topic
corpus_lsi = lsi[corpus_tfidf]

# =============================================================================
#for doc in corpus_lsi[:10]:
#     print(doc)
# =============================================================================
query = "good dress"

# Change query word to vector
query_bow = dct.doc2bow(query.lower().split())
print(query_bow)

# Mapping query word to 100 dimensional topic space with LSI model
query_lsi = lsi[query_bow]
for idx, val in enumerate(query_lsi):
        print(val[0], ',', val[1])

# Calculate the cosine similarity/correlation degree btw documents and query word
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[query_lsi]
print(list(enumerate(sims)))
# Output sorted results
sort_sims = sorted(enumerate(sims), key = lambda item:-item[-1])
top10 = sort_sims[:10]
top10doc = [texts[j[0]] for j in top10]

print(top10doc)

############################
# Train lda model
lda = models.LdaModel(corpus_tfidf, id2word = dct, num_topics = 100)

# Compute Perplexity
print('\nPerplexity: ', lda.log_perplexity(corpus_tfidf))  
# a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = models.CoherenceModel(model=lda, texts=texts, dictionary=dct, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# =============================================================================
# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus_tfidf, dct)
# vis
# =============================================================================


###########################
# Cross-validation of LDA
sentiment = df['Recommended IND']
#
f, axes = plt.subplots(1,1, figsize=(14,4), sharex=False)
sns.countplot(x='Recommended IND', data=df,order=sentiment.value_counts().index)
axes.set_title("Frequency Distribution for\nRecommended IND")
axes.set_ylabel("Occurrence")
axes.set_xlabel('Recommended IND')
fig = plt.gcf()
fig.savefig(os.path.join(path_save_fig,'RecommendedIND.png'))
plt.show()

#
print('Cross-Validation...')
t1 = time.time()
[best_topic, best_lda, cv_errors] = ldaCV(texts, sentiment, n_topics = np.linspace(5,100,20).astype(int))
print('Time used: %s seconds' % (time.time()-t1))
# Save model to disk
best_lda.save(os.path.join(path_save_data,'best_lda'))
# =============================================================================
# # Load a potentially pretrained model from disk.
# best_lda = LdaModel.load(os.path.join(path_save_data,'best_lda'))
# =============================================================================

# Svae figure
plt.figure()
plt.plot(np.linspace(5,100,20).astype(int), cv_errors, linewidth = 6)
plt.axvline(x = best_topic, color = 'r', linestyle = '--')
plt.xlabel('# of Topics', fontsize = 18)
plt.ylabel('CV Error', fontsize = 18)
plt.title('Cross-validation errors for LDA', fontsize = 18)
fig = plt.gcf()
fig.set_size_inches(18.5,10.5,5)
fig.savefig(os.path.join(path_save_fig, 'cv_errors.png'))

#######################
# Gram Table
# Recommended
Recommended_gramtab = gram_table(data= documents[df["Recommended IND"].astype(int) == 0], gram=[1,2,3], length=20)
Recommended_gramtab.to_csv(os.path.join(path_save_data, 'Recomended_Gram_Tab.csv'))
#
NotRecommended_gramtab = gram_table(data= documents[df["Recommended IND"].astype(int) == 1], gram=[1,2,3], length=20)
NotRecommended_gramtab.to_csv(os.path.join(path_save_data, 'NotRecomended_Gram_Tab.csv'))


##########################################################
# Creating Bigrams and Trigrams Models, higher threshold fewer phrases
##########################################################
print('Creating Bigrams Model...')
# Build the bigram models
t1 = time.time()
bigram = models.Phrases(texts, min_count=1, threshold=1) 
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = models.phrases.Phraser(bigram)
print('Time used: %s seconds' % (time.time()-t1))
# See example
print(bigram_mod[texts[9]])
texts_bigram = bigram_mod[texts]
#####################
# save
with open(os.path.join(path_save_data,'Preprocessed Review Text Bigram.txt'), 'w') as f:
    for text in texts_bigram:
        for item in text:
            f.write("%s " % item)
            
        f.write("\n")
# Read

# =============================================================================
# texts_bigram = []
# with open(os.path.join(path_save_data,'Preprocessed Review Text Bigram.txt'), 'r') as f:
#     for line in f:
#         text = line.split()
#         texts_bigram.append(text)
# =============================================================================
#
print('Cross-Validation...')
t1 = time.time()
n_topics = np.linspace(5,100,20).astype(int)
[best_topic_bi, best_lda_bi, cv_errors_bi] = ldaCV(texts_bigram, sentiment, n_topics)
print('Time used: %s seconds' % (time.time()-t1))
# Save model to disk
best_lda.save(os.path.join(path_save_data,'best_lda_bigram'))

# Svae figure
plt.figure()
plt.plot(n_topics, cv_errors_bi, linewidth = 6)
plt.axvline(x = best_topic_bi, color = 'r', linestyle = '--')
plt.xlabel('# of Topics', fontsize = 18)
plt.ylabel('CV Error', fontsize = 18)
plt.title('Cross-validation errors for LDA-Bigram', fontsize = 18)
fig = plt.gcf()
fig.set_size_inches(18.5,10.5,5)
plt.show()
fig.savefig(os.path.join(path_save_fig, 'cv_errors_bigram.png'))
#######################
# Build the trigram models
t1 = time.time()
trigram = models.Phrases(bigram[texts], threshold=1) 
# Faster way to get a sentence clubbed as a trigram/bigram
trigram_mod = models.phrases.Phraser(trigram) 
print('Time used: %s seconds' % (time.time()-t1))
# See example
print(trigram_mod[bigram_mod[texts[9]]])
texts_trigram = trigram_mod[bigram_mod[texts]]
#####################
# save
with open(os.path.join(path_save_data,'Preprocessed Review Text Trigram.txt'), 'w') as f:
    for text in texts_trigram:
        for item in text:
            f.write("%s " % item)
            
        f.write("\n")
# Read

# =============================================================================
# texts_trigram = []
# with open(os.path.join(path_save_data,'Preprocessed Review Text Trigram.txt'), 'r') as f:
#     for line in f:
#         text = line.split()
#         texts_trigram.append(text)
# =============================================================================
#
print('Cross-Validation...')
t1 = time.time()
n_topics = np.linspace(5,100,20).astype(int)
[best_topic_tri, best_lda_tri, cv_errors_tri] = ldaCV(texts_trigram, sentiment, n_topics)
print('Time used: %s seconds' % (time.time()-t1))
# Save model to disk
best_lda.save(os.path.join(path_save_data,'best_lda_trigram'))
# Svae figure
plt.figure()
plt.plot(n_topics, cv_errors_tri, linewidth = 6)
plt.axvline(x = best_topic_tri, color = 'r', linestyle = '--')
plt.xlabel('# of Topics', fontsize = 18)
plt.ylabel('CV Error', fontsize = 18)
plt.title('Cross-validation errors for LDA-Trigram', fontsize = 18)
fig = plt.gcf()
fig.set_size_inches(18.5,10.5,5)
plt.show()
fig.savefig(os.path.join(path_save_fig, 'cv_errors_trigram.png'))


#############################
# Using scipy.sparse.csr.csr_matrix to get unigram, bigram and trigram textual features
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(texts)
X = vect.transform(texts)
y = df["Recommended IND"].copy()
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=23, stratify=y)

model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

print("Train Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_train), y_train)))
print("Train Set ROC: {}\n".format(metrics.roc_auc_score(model.predict(X_train), y_train)))

print("Validation Set Accuracy: {}".format(metrics.accuracy_score(model.predict(X_valid), y_valid)))
print("Validation Set ROC: {}".format(metrics.roc_auc_score(model.predict(X_valid), y_valid)))

target_names = ["Not Recommended","Recommended"]
from IPython.display import display, HTML, Image
display(eli5.show_weights(model, vec=vect, top=100,
                  target_names=target_names))