#!/usr/bin/env python
# coding: utf-8

# In[2]:
"""
from IPython import get_ipython

get_ipython().system('pip install -q wordcloud')
"""
import wordcloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import unicodedata
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn import metrics
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS  # a longer list of stopwords than nltk

pd.set_option("display.precision", 4)
np.set_printoptions(precision=4)




# In[3]:


from sklearn.datasets import fetch_20newsgroups
CATEGORIES = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

IS_TRAIN_SET = False  # get the test set so that we can compare performance to the one documented on the above scikit web page

if IS_TRAIN_SET:
  news_docs = fetch_20newsgroups(subset='train', categories=CATEGORIES, shuffle=True, random_state=42)
else:
  news_docs = fetch_20newsgroups(subset='test', categories=CATEGORIES, shuffle=True, random_state=42)

TARGET_NAMES = news_docs.target_names
print("TARGET_NAMES:", TARGET_NAMES)
print("len(news_docs.data):", len(news_docs.data))

print(type(news_docs))
data_samples = news_docs.data
data_classes = news_docs.target
print("data_classes:", data_classes)

n_topic_clusters = np.unique(data_classes).shape[0]
print("n_topic_clusters:", n_topic_clusters)

print("Class distribution:")
print(pd.Series(data_classes).value_counts())


# In[4]:


#@title
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 

# Get stopwords, stemmer and lemmatizer
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
general_stopwords = ENGLISH_STOP_WORDS  # note: sklearn has a longer list than nltk.corpus.stopwords.words('english')
domain_stopwords = ['subject', 'organization', 'lines', 'from', 'reply-to', 'distribution', 'keywords', 'article', 'newsreader', 'nntp-posting-host', 'writes', 'x-newsreader', 'summary']


# In[5]:


def remove_accents(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters or x == " ")
  
def remove_misc(data):
  return data.replace('\n', '')


# In[6]:


def preprocess_documents(data_samples, data_classes, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True, is_print=False, is_plot=False):
  '''
  Preprocess documents
  
  Parameters:
  data_samples: list of data samples
  data_classes: list of classes
  general_stopwords: list of stopwords used in general
  domain_stopwords: domain specific stopwords that should not appear in the word cloud
  stemmer: the stemmer method
  lemmatizer: the lemmatizer method
  is_skip_header: boolean to skip the header section in each newsgroup
  
  Return:
  df_token_lists: dataframe of tokenized word lists
  df_lem_strings: dataframe of lemmatized word lists
  data_samples_processed: list of the processed lemmatized strings
  data_processed: list of tuples of data_samples_processed and their data classes
  '''

  # POS (Parts Of Speech)
  # For: nouns, adjectives, verbs and adverbs use {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'} 
  DI_POS_TYPES = {'NN':'n'}  # Just the nouns
  POS_TYPES = list(DI_POS_TYPES.keys())

  # Constraints on tokens
  MIN_STR_LEN = 3
  MAX_TOKENS = 1200
  RE_VALID = '[a-zA-Z]'
  PUNCT = ['@', '_', '.']

  # Process all data samples
  li_tokens = []
  li_token_lists = []
  li_lem_strings = []
  len_tokens = []
  
  for i,text in enumerate(data_samples):

      # Tokenize by sentence, then by lowercase word
      tokens = []
      if is_skip_header:
        # Skip most of the first section that represents the header metadata
        for j,sent in enumerate(nltk.sent_tokenize(text)):
          if j == 0:
            li_sents = [s.strip() for s in sent.splitlines()]
            li_sents = [s for s in li_sents if len(s) > 0 and not s.lower().startswith(tuple(domain_stopwords))]
            tokens0 = [token.lower() for sent in li_sents for token in nltk.word_tokenize(sent)]
            tokens = []
            for t in tokens0:
              is_in = False
              for p in PUNCT:
                if p in t:
                  is_in = True
                  break
                  
              if not is_in:    
                tokens.append(t)
              
            continue

          for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
      else:
        tokens = [token.lower() for sent in nltk.sent_tokenize(text) for token in nltk.word_tokenize(sent)]
      
      # Remove stopwords
      tokens = [x for x in tokens if x not in general_stopwords]
      tokens = [x for x in tokens if x not in domain_stopwords] 
      
      # Limit
      tokens = tokens[:MAX_TOKENS]
      len_tokens.append(len(tokens))

      # Process all tokens per quote
      li_tokens_quote = []
      li_tokens_quote_lem = []
      for token in tokens:
          # Remove email addresses
          if '@' in token:
            continue
          
          # Remove accents
          t = remove_accents(token)

          # Remove misc
          t = remove_misc(t)

          # Remove punctuation
          t = str(t).translate(string.punctuation)
          li_tokens_quote.append(t)

          # Add token that represents "no lemmatization match"
          li_tokens_quote_lem.append('-') # this token will be removed if a lemmatization match is found below

          # Process each token
          if t not in general_stopwords:
              if re.search(RE_VALID, t):
                  if len(t) >= MIN_STR_LEN:
                      # Note that the POS (Part Of Speech) is necessary as input to the lemmatizer 
                      # (otherwise it assumes the word is a noun)
                      pos = nltk.pos_tag([t])[0][1][:2]
                      pos2 = 'n'  # set default to noun
                      if pos in DI_POS_TYPES:
                        pos2 = DI_POS_TYPES[pos]

                      stem = stemmer.stem(t)
                      lem = lemmatizer.lemmatize(t, pos=pos2)  # lemmatize with the correct POS

                      if pos in POS_TYPES:
                          li_tokens.append((t, stem, lem, pos))

                          # Remove the '-' token and append the lemmatization match
                          li_tokens_quote_lem = li_tokens_quote_lem[:-1] 
                          li_tokens_quote_lem.append(lem)

      # Build list of token lists from lemmatized tokens
      li_token_lists.append(li_tokens_quote)

      # Build list of strings from lemmatized tokens
      str_li_tokens_quote_lem = ' '.join(li_tokens_quote_lem)
      li_lem_strings.append(str_li_tokens_quote_lem)

  # Build resulting dataframes from lists
  df_token_lists = pd.DataFrame(li_token_lists)

  # Replace None with empty string
  for c in df_token_lists:
      if str(df_token_lists[c].dtype) in ('object', 'string_', 'unicode_'):
          df_token_lists[c].fillna(value='', inplace=True)

  df_lem_strings = pd.DataFrame(li_lem_strings, columns=['lem string'])

  # Build a dataset that corresponds the processed strings to the class labels
  data_samples_processed = [x.replace('-','') for x in li_lem_strings]
  data_processed = list(zip(data_classes, data_samples_processed))
  
  if is_print:
    print("---")
    print("Before removing POS that are not nouns and lemmatization:")
    print("df_token_lists.shape:", df_token_lists.shape)
    print("df_token_lists.head(5):")
    print(df_token_lists.head(5).to_string())
    
    print("---")
    print("After removing POS that are not nouns and lemmatization:")
    print("df_lem_strings.head():")
    print(df_lem_strings.head().to_string())
    
    print("---")
    for i,v in enumerate(data_processed[:5]):
      print(i,v)

  # Plot histogram of tokenized document lengths
  if is_plot:
    fig = plt.figure(figsize=(8,6))
    plt.suptitle("Histogram of tokenized document lengths")
    plt.xlabel("Document lengths")
    plt.ylabel("Counts")
    n, bins, patches = plt.hist(len_tokens, bins=50, density=False, facecolor='g', alpha=0.75)
    plt.show()
  
  return df_token_lists, df_lem_strings, data_samples_processed, data_processed

# Preprocess documents
df_token_lists, df_lem_strings, data_samples_processed, data_processed =   preprocess_documents(data_samples, data_classes, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True, is_print=True, is_plot=True)


# In[7]:


# Build tf-idf vectorizer and related variables from the input documents
# to support both unigrams and bigrams use: ngram_range=(1,2)
def build_vectorizer(documents, ngram_range=(1,1), max_df=1.0, min_df=1, stop_words=None, max_features=None):   # default CountVectorizer parameter values
   '''
   (i) Build count_vectorizer from the documents and fit the documents  
   (ii) Build TF (Term Frequency) from the documents, this is a sparse version of the bag-of-words  
   (iii) Build bag-of-words in two steps: fit, transform  
   (iv) Get feature names and build dataframe version of the bag-of-words  
   (v) Use TfidfTransformer to transform bag_of_words into TF-IDF matrix (Term Frequency - Inverse Document Frequency)  
   (vi) Find most popular words and highest weights  
   (vii) Build word weights as a list and sort them  
   (viii) Calculate cosine similarity of all documents with themselves  
   (ix) Calculate distance matrix of documents  
   
   Note:
   The TF_IDF matrix can be built directly with 'TfidfVectorizer' instead of using 'CountVectorizer' followed by 'TfidfTransformer'
   
   Return:
   cvec: CountVectorizer
   tf: Term Frequencies
   tfidf: TF-IDF matrix 
   feature_names: Feature names in TF-IDF
   df_bag_of_words: Bag of words from the sparse Term Frequencies
   df_weights: Most popular words, word counts and highest weights
   cos_sim: Cosine similarity of all documents with themselves
   samp_dist: Distance matrix of documents
   '''
   
   # Build CountVectorizer from the documents and fit the documents
   count_vectorizer = CountVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, stop_words=stop_words, max_features=max_features)
   
   # Build TF (Term Frequencies) from the documents, this is a sparse version of the bag-of-words
   tf = count_vectorizer.fit_transform(documents)  # note that the LDA transform needs this for its clustering model (explained below)

   # Build bag-of-words in two steps: fit, transform
   cvec = count_vectorizer.fit(documents)
   bag_of_words = cvec.transform(documents)
   
   # Get feature names and build dataframe version of the bag-of-words
   feature_names = cvec.get_feature_names()
   df_bag_of_words = pd.DataFrame(bag_of_words.todense(), columns=feature_names)
   
   # Use TfidfTransformer to transform bag_of_words into TF-IDF matrix (Term Frequency - Inverse Document Frequency)
   transformer = TfidfTransformer()
   tfidf = transformer.fit_transform(bag_of_words)

   # Find most popular words, word counts and highest weights
   word_cnts = np.asarray(bag_of_words.sum(axis=0)).ravel().tolist()  # for each word in column, sum all row counts
   df_cnts = pd.DataFrame({'word': feature_names, 'count': word_cnts})
   df_cnts = df_cnts.sort_values('count', ascending=False)

   # Build word weights as a list and sort them (the dataframe below also contains the counts)
   weights = np.asarray(tfidf.mean(axis=0)).ravel().tolist()
   df_weights = pd.DataFrame({'word': feature_names, 'weight': weights})
   df_weights = df_weights.sort_values('weight', ascending=False)

   df_weights = df_weights.merge(df_cnts, on='word', how='left')
   df_weights = df_weights[['word', 'count', 'weight']]

   # Calc cosine similarity of all documents with themselves
   cos_sim = metrics.pairwise.cosine_similarity(tfidf, tfidf)

   # Calc distance matrix of documents
   samp_dist = 1 - cos_sim

   return cvec, tf, tfidf, feature_names, df_bag_of_words, df_weights, cos_sim, samp_dist
 
# Build TF-IDF matrices
li_lem_strings = df_lem_strings['lem string'].values.tolist()
cvec, tf, tfidf, feature_names, df_bag_of_words, df_weights, cos_sim, samp_dist = build_vectorizer(li_lem_strings)


# In[8]:

"""
def word_cloud(df_weights, n_top_words=10, is_print=True, is_plot=True):
  '''
  Build a word cloud
  '''
  s_word_freq = pd.Series(df_weights['count'])
  s_word_freq.index = df_weights['word']
  di_word_freq = s_word_freq.to_dict()
  cloud = wordcloud.WordCloud(width=900, height=500).generate_from_frequencies(di_word_freq)
 
  if is_print:
    print(df_weights.iloc[:n_top_words,:])
  
  if is_plot:
    plt.imshow(cloud)
    plt.axis('off')
    #plt.show()
  
  return cloud
  
# Build word cloud
print("Word cloud based on all categories:")
cloud_all = word_cloud(df_weights, is_print=True)


# In[9]:


def word_cloud_per_class(i_class, data_samples, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True):
  print("Category:", TARGET_NAMES[i_class])
  print("----------------------------------")
  data_samples_group = [x for i,x in enumerate(data_samples) if data_classes[i] == i_class]
  data_classes_group = [i_class] * len(data_samples_group)
  
  # Preprocess documents
  df_token_lists_per_class, df_lem_strings_per_class, data_samples_processed_per_class, data_processed_per_class =     preprocess_documents(data_samples_group, data_classes_group, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header)
  
  # Build TF-IDF matrices
  li_lem_strings = df_lem_strings_per_class['lem string'].values.tolist()
  cvec, tf, tfidf, feature_names, df_bag_of_words, df_weights, cos_sim, samp_dist =     build_vectorizer(li_lem_strings)

  # Build word cloud
  cloud = word_cloud(df_weights)
  return cloud


# In[ ]:


i_class = 0
word_cloud_per_class(i_class, data_samples, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True)


# In[ ]:


i_class = 1
word_cloud_per_class(i_class, data_samples, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True)


# In[ ]:


i_class = 2
word_cloud_per_class(i_class, data_samples, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True)


# In[ ]:


i_class = 3
word_cloud_per_class(i_class, data_samples, general_stopwords, domain_stopwords, stemmer, lemmatizer, is_skip_header=True)


# In[ ]:
"""

"""
# Dimensionality reduction using PCA
# Reduce the tfidf matrix to just 2 features (n_components)
X = tfidf.todense()
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

print("X_pca now has just 2 columns:")
print(X_pca[:5,:])

print("---")
print("X.shape:", np.array(X).shape)
print("X_pca.shape:", np.array(X_pca).shape)
"""
#################PSO##################

import numpy as np
import random
import math
import os
import sys

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
  a = 0.00001
  diff = 0
  avg = sum(x)/float(len(x))
  for i in range (0,len(x)):
    if x[i]==-1 or x[i]==0:
      t=0
    else:
      t=1
    diff = diff+t-avg
    if x[i]>0:
      a = a+1

  total = diff/a 
  return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
  def __init__(self,x0):
    self.position_i=[]          # particle position
    self.velocity_i=[]          # particle velocity
    self.pos_best_i=[]          # best position individual
    self.err_best_i=0.1          # best error individual
    self.err_i=0.1               # error individual

    for i in range(0,num_dimensions):
      self.velocity_i.append(random.uniform(-1,1))
      self.position_i.append(x0[i])

  # evaluate current fitness
  def evaluate(self,costFunc):
    self.err_i=costFunc(self.position_i)

  # check to see if the current position is an individual best
    if self.err_i > self.err_best_i or self.err_best_i==0.1:
      self.pos_best_i=list(self.position_i)
      self.err_best_i=self.err_i
                    
  # update new particle velocity
  def update_velocity(self,pos_best_g):
    w=0.5       # constant inertia weight (how much to weigh the previous velocity)
    c1=1        # cognative constant
    c2=2        # social constant
        
    for i in range(0,num_dimensions):
      r1=random.random()
      r2=random.random()
            
      vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
      vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
      self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

  # update the particle position based off new velocity updates
  def update_position(self):
    for i in range(0,num_dimensions):
      s=0
      lim = 1.0/(1+ math.exp(-(self.velocity_i[i])))
      if random.uniform(0,1) < lim :
        s = 1
      if random.uniform(0,1) > s and self.position_i[i]>-1 :  
        self.position_i[i] = 0
            
    
class PSO():
  def __init__(self, costFunc, x0,num_particles, maxiter, verbose=False):
    global num_dimensions
    
    global X_PSO
    X_PSO=[]
    
    err_best_g=0.1                   # best error for group
    pos_best_g=[]                   # best position for group

    # establish the swarm
    swarm=[]
    #print num_particles
    for i in range(0,num_particles):
      num_dimensions=len(x0[i])
      #print x0[i]
      swarm.append(Particle(x0[i]))
      #print swarm[i].position_i

    # begin optimization loop
    i=0
    while i<maxiter:
      if verbose: 
        print('iter:',i, 'best solution:',err_best_g)
      # cycle through particles in swarm and evaluate fitness
      for j in range(0,num_particles):
        swarm[j].evaluate(costFunc)

      # determine if current particle is the best (globally)
        if swarm[j].err_i>err_best_g or err_best_g==0.1:
          pos_best_g=list(swarm[j].position_i)
          err_best_g=float(swarm[j].err_i)
            
      # cycle through swarm and update velocities and position
      for j in range(0,num_particles):
        swarm[j].update_velocity(pos_best_g)
        swarm[j].update_position()
        #print swarm[j].position_i
      i+=1

    # print final results
    print("###################### PSO ####################")
    print('\nFINAL SOLUTION:')
    print(pos_best_g)
    print(err_best_g)
    for i in range(0,num_particles):
      X_PSO.append(swarm[i].position_i)
    #return X_PSO

if __name__ == "__PSO__":
  main()

#--- RUN ----------------------------------------------------------------------+

#initial=[[5,4,3],[1,2,0]]               # initial starting location [x1,x2...]
#bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]

X = tfidf.todense()
X=X.tolist()

#print(type(X))
#print(X.shape)
#print (X)
for item in X:
  #print (item)
  for i in range(0,len(item)):
    if item[i] == 0:
      item[i] = -1
  
#print (initial)

PSO(func1, X, num_particles=len(X), maxiter=3, verbose=True)
print("#################X_PSO#################")
print (X_PSO)

#--- END ----------------------------------------------------------------------+

######################################


# In[ ]:


# Note that for this simplistic dataset we don't really need to iterate
# and usually we don't know the number of topic clusters in advance!
km_model = KMeans(n_clusters=n_topic_clusters, max_iter=10, n_init=2, random_state=0)

# K-means (transform dimensions from number of features in input matrix to n_clusters)
km_model.fit(X_PSO)

print("Cluster Centers")
print(km_model.cluster_centers_)
#print((km_model.cluster_centers_).shape)

t=km_model.cluster_centers_

#print (type(t))
#print (t.shape)
t=t[:,0:2]
#print (t.shape, type(t))
df_centers = pd.DataFrame(t, columns=['x', 'y'])

"""
print("df_centers:")
print(df_centers)
"""

# In[ ]:


# K-means model labels and actual document labels
df_documents = pd.DataFrame(data_processed, columns=['Label', 'Document'])
#print(df_documents.head())


km_model_labels = km_model.labels_.tolist()

#################ACTUAL LABELS########################
df_documents = pd.DataFrame(data_processed, columns=['Label', 'Document'])
#print(df_documents.head())
actual_labels = df_documents['Label'].tolist()

"""
print (set(km_model_labels))
for i in range(0,len(km_model_labels)):
  print(km_model_labels[i], actual_labels[i])


print("Assigned Label, Actual Label")
for i in range(0,len(km_model_labels)):
  print(km_model_labels[i], actual_labels[i])
"""

# In[ ]:
X_PSO=np.asarray(X_PSO)

def scatter_plot_with_labels(i_plot, df_centers, labels, title):
  ax = plt.subplot(1, 2, i_plot)
  ax.set_title('PCA features colored by class for:\n' + title)
  ax.set_xlabel('x distance')
  ax.set_ylabel('y distance')
  print (type(X_PSO))
  print (len(X_PSO))
  #print (X_PSO[0])
  plt.scatter(X_PSO[:,0], X_PSO[:,1], c=labels, s=50, cmap='gray')

  colors = ['red', 'green', 'blue', 'yellow']
  for i,color in enumerate(df_centers.index.tolist()):
    plt.plot(df_centers['x'][i], df_centers['y'][i], 'X', label='K-means center: %d' % i, color=colors[i])

  plt.legend()

plt.figure(figsize=(12,6))
scatter_plot_with_labels(1, df_centers, km_model_labels, "km_model_labels")
scatter_plot_with_labels(2, df_centers, actual_labels, "actual_labels")
plt.show()


# In[ ]:


# di_actual_to_kmeans = {0:0, 1:1, 2:2, 3:3}  # no color remapping
di_actual_to_kmeans = {0:3, 1:2, 2:1, 3:0}
km_model_labels_remapped = [di_actual_to_kmeans[x] for x in km_model_labels]

plt.figure(figsize=(12,6))
scatter_plot_with_labels(1, df_centers, km_model_labels_remapped, "km_model_labels_remapped")
scatter_plot_with_labels(2, df_centers, actual_labels, "actual_labels")
plt.show()


# In[ ]:





# In[ ]:




