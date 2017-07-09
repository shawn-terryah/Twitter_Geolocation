import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# Tokenizer to use for text vectorization
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

def tokenize(tweet):
    return tknzr.tokenize(tweet)

'''STEP 1: PREPARE THE CLEANED DATA FOR MODEL FITTING'''

# Read cleaned_training_tweets into pandas and randomize it
df = pd.read_pickle('cleaned_training_tweets.pkl')
randomized_df = df.sample(frac=1, random_state=111)

# Split randomized_df into two disjoint sets 
base_df = randomized_df.iloc[:750000, :]      # used to train the base classifiers
meta_df = randomized_df.iloc[750000:, :]      # used to train the meta classifier

# Known geotagged location labels to use for model fitting
base_y = base_df['closest_major_city'].values
meta_y = meta_df['closest_major_city'].values

'''STEP 2: TRAIN A BASE-LEVEL LINEAR SVC CLASSSIFIER ON THE USER DESCRIBED LOCATIONS'''

# Raw text of user described locations
base_location_doc = base_df['user_described_location'].values
meta_location_doc = meta_df['user_described_location'].values

# Fit a tf-idf vectorizer using base_location_doc and use it to transform base_location_doc & meta_location_doc
location_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
base_location_X = location_vect.fit_transform(base_location_doc.ravel())
meta_location_X = location_vect.transform(meta_location_doc)

# Fit a Linear SVC Model with 'base_location_X' and 'base_y'. Note: it is important to use 
# balanced class weights otherwise the model will overwhelmingly favor the majority class.
location_SVC = LinearSVC(class_weight='balanced')
location_SVC.fit(base_location_X, base_y)

# We can now pass meta_location_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
location_SVC_decsfunc = location_SVC.decision_function(meta_location_X)

# Pickle the location vectorizer and the linear SVC model for future use
joblib.dump(location_vectorizer, 'USER_LOCATION_VECTORIZER.pkl')
joblib.dump(location_SVC, 'USER_LOCATION_SVC.pkl')


'''STEP 3: TRAIN A BASE-LEVEL LINEAR SVC CLASSIFIER ON THE TWEETS'''

# Raw text of tweets
base_tweet_doc = base_df['tweet'].values
meta_tweet_doc = meta_df['tweet'].values

# Fit a tf-idf vectorizer using base_tweet_doc and use it to transform base_tweet_doc & meta_tweet_doc
tweet_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
base_tweet_X = tweet_vectorizer.fit_transform(base_tweet_doc.ravel())
meta_tweet_X = tweet_vectorizer.transform(meta_tweet_doc)

# Fit a Linear SVC Model with 'base_tweet_X' and 'base_tweet_y'. Note: it is important to use 
# balanced class weights otherwise the model will overwhelmingly favor the majority class.
tweet_SVC = LinearSVC(class_weight='balanced')
tweet_SVC.fit(base_tweet_X, base_y)

# We can now pass meta_tweet_X into the fitted model and save the decision 
# function, which will be used in Step 4 when we train the meta random forest
tweet_SVC_decsfunc = tweet_SVC.decision_function(meta_tweet_X)

# Pickle the tweet vectorizer and the linear SVC model for future use
joblib.dump(tweet_vectorizer, 'TWEET_VECTORIZER.pkl')
joblib.dump(tweet_SVC, 'TWEET_SVC.pkl')


'''STEP 4: TRAIN A META-LEVEL RANDOM FOREST CLASSIFIER'''

# additional features from meta_df to pull into the final model
friends_count = meta_df['friends_count'].values.reshape(741105, 1)
utc_offset = meta_df['utc_offset'].values.reshape(741105, 1)
tweet_time_secs = meta_df['tweet_time_secs'].values.reshape(741105, 1)
statuses_count = meta_df['statuses_count'].values.reshape(741105, 1)
favourites_count = meta_df['favourites_count'].values.reshape(741105, 1)
followers_count = meta_df['followers_count'].values.reshape(741105, 1)
listed_count = meta_df['listed_count'].values.reshape(741105, 1)

# np.hstack these additional features together
add_features = np.hstack((friends_count, 
                          utc_offset, 
                          tweet_time_secs,
                          statuses_count,
                          favourites_count,
                          followers_count,
                          listed_count))

# np.hstack add_features with the two decision function varibles from steps 2 & 3 
meta_X = np.hstack((location_SVC_decsfunc,        # from Step 2
                    tweet_SVC_decsfunc,           # from Step 3 
                    add_features))

# Fit Random Forest with 'meta_X' and 'meta_y'
meta_RF = RandomForestClassifier(n_estimators=60, n_jobs=-1)
meta_RF.fit(meta_X, meta_y)

# Pickle the meta Random Forest for future use
joblib.dump(meta_RF, 'META_RF.pkl')
