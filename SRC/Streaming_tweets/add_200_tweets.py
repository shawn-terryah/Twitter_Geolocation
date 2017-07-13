import tweepy
import pandas as pd

# these are provided to you through the Twitter API after you create a account
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

count = 0

def get_200_tweets(screen_name):
    '''
    Helper function to return a list of a Twitter user's 200 most recent tweets
    '''
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth,
                     wait_on_rate_limit=True,    
                     wait_on_rate_limit_notify=True) 

    # Initialize a list to hold the user's 200 most recent tweets
    tweets_data = []
    
    global count
    
    try:
        # make request for most recent tweets (200 is the maximum allowed per distinct request)
        recent_tweets = api.user_timeline(screen_name = screen_name, count=200)

        # save data from most recent tweets
        tweets_data.extend(recent_tweets)
        
        count += 1
        
        # Status update
        if count % 1000 == 0:
            print 'Just stored tweets for user #{}'.format(count)
            
    except:
        count += 1
        pass
    
    # pull only the tweets and encode them in utf-8 to preserve emojis
    list_of_recent_tweets = [''.join(tweet.text.encode("utf-8")) for tweet in tweets_data]
    
    return list_of_recent_tweets

# Create a new column in evaluation_df called '200_tweets'
evaluation_df['200_tweets'] = map(lambda x: get_200_tweets(x), evaluation_df['screen_name'])
