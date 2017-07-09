import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Data to fit the base Linear SVC model
base_tweet_doc = base_df['tweet'].values
base_tweet_Y = base_df['closest_major_city'].values

# Data to fit the meta Random Forest model
meta_tweet_doc = meta_df['tweet'].values
meta_location_Y = meta_df['closest_major_city'].values

# Tokenize and vectorize the user described locations
tweet_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
base_tweet_X = tweet_vectorizer.fit_transform(base_tweet_doc.ravel())

# Use tweet_vectorizer to transform meta_tweet_doc
meta_tweet_X = tweet_vectorizer.transform(meta_tweet_doc)

# Fit a Linear SVC Model with 'base_tweet_X' and 'base_tweet_Y'. 
# It is important to use balanced class weights otherwise the model will
# overwhelmingly favor the majority class.
tweet_SVC = LinearSVC(class_weight='balanced')
tweet_SVC.fit(base_tweet_X, base_tweet_Y)

# We can now pass meta_tweet_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
tweet_SVC_decsfunc = tweet_SVC.decision_function(meta_tweet_X)

# Pickle the tweet vectorizer and the linear SVC model for future use
joblib.dump(tweet_vect, 'TWEET_VECTORIZER.pkl')
joblib.dump(tweet_SVC, 'TWEET_SVC.pkl')
