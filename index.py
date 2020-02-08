# dataset link https://www.kaggle.com/subhassing/exploring-consumer-complaint-data/data

# Step 1 Import the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing,linear_model, naive_bayes, metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import os
from textblob import TextBlob
from textblob import Word

from io import StringIO
import seaborn as sns


# Step 2 Importing the data

Data = pd.read_csv("consumer_complaints.csv",encoding='latin-1')

# Step 3 Data understanding

# Selecting required columns and rows
Data = Data[['product', 'consumer_complaint_narrative']]
Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]

# See top 5 rows
Data.head()

# Factorizing the category column
Data['category_id'] = Data['product'].factorize()[0]
Data.head()

# Check the distriution of complaints by category
Data.groupby('product').consumer_complaint_narrative.count()


# Lets plot it and see
fig = plt.figure(figsize=(8,6))
Data.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()

# Step 4 Splitting the data


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Data['consumer_complaint_narrative'], Data['product'])

# Step 5 Feature engineering using TF-IDF

encoder = preprocessing.LabelEncoder() # Encode labels with value between 0 and n_classes-1.
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


tfidf_vect = TfidfVectorizer(analyzer='word',token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Data['consumer_complaint_narrative'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# Step 6 Model building and evaluation

'''

Suppose we are building a linear classifier on word-level TF-IDF vectors.
We are using default hyper parameters for the classifier. Parameters can be
changed like C, max_iter, or solver to obtain better results.

'''

model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)


accuracy = metrics.accuracy_score(model.predict(xvalid_tfidf),valid_y)
print ("Accuracy: ", accuracy)

# Classification report

print(metrics.classification_report(valid_y, model.predict(xvalid_tfidf),target_names=Data['product'].unique()))

#confusion matrix
conf_mat = confusion_matrix(valid_y, model.predict(xvalid_tfidf))

# Vizualizing confusion matrix

category_id_df = Data[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','product']].values)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="BuPu",xticklabels=category_id_df[['product']].values,yticklabels=category_id_df[['product']].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# Step 7 prediction

# Prediction example
texts = ["This company refuses to provide me verification and validation of debt per my right under the FDCPA.I do not believe this debt is mine."]
text_features = tfidf_vect.transform(texts)
predictions = model.predict(text_features)
print(texts)
print("  - Predicted as: '{}'".format(id_to_category[predictions[0]]))




# -------------------------------------------------------------------------

'''

To increase the accuracy, we can do the following things:

• Reiterate the process with different algorithms like
Random Forest, SVM, GBM, Neural Networks, Naive
Bayes.
• Deep learning techniques like RNN and LSTM can also be used.

• In each of these algorithms, there are so many
parameters to be tuned to get better results. It can be
easily done through Grid search, which will basically
try out all possible combinations and give the best out.

'''






