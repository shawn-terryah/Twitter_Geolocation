import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Data to fit the base Linear SVC model
base_location_doc = base_df['user_described_location'].values
base_location_Y = base_df['closest_major_city'].values

# Data to fit the meta Random Forest model
meta_location_doc = meta_df['user_described_location'].values
meta_location_Y = meta_df['closest_major_city'].values

# Tokenize and vectorize the user described locations
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

def tokenize(tweet):
    return tknzr.tokenize(tweet)
  
location_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
base_location_X = location_vect.fit_transform(base_location_doc.ravel())

# Use location_vectorizer to transform the meta_location_doc
meta_location_X = location_vect.transform(meta_location_doc)

# Fit a Linear SVC Model with 'base_location_X' and 'base_location_Y'. 
# It is important to use balanced class weights otherwise the model will
# overwhelmingly favor the majority class.
location_SVC = LinearSVC(class_weight='balanced')
location_SVC.fit(base_location_X, base_location_Y)

# We can now pass meta_location_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
location_SVC_decsfunc = location_SVC.decision_function(meta_location_X)

# Pickle the location vectorizer and the linear SVC model for future use
joblib.dump(location_vectorizer, 'USER_LOCATION_VECTORIZER.pkl')
joblib.dump(location_SVC, 'USER_LOCATION_SVC.pkl') 
