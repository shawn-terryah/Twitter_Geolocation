import tweepy
import json
from pymongo import MongoClient

class StreamListener(tweepy.StreamListener):
    """
    tweepy.StreamListener is a class provided by tweepy used to access
    the Twitter Streaming API. It allows us to retrieve tweets in real time.
    """

    def on_connect(self):
        """Called when the connection is made"""
        print("You're connected to the streaming server.")

    def on_error(self, status_code):
        """This is called when an error occurs"""
        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):
        """This will be called each time we receive stream data"""
        client = MongoClient()

        # Store the tweet data in a database callled 'training_tweets' in MongoDB, if 
        # 'training_tweets' does not already exist it will be created for you
        db = client.training_tweets

        # Decode JSON
        datajson = json.loads(data)

        # I'm only storing tweets in English
        if "lang" in datajson and datajson["lang"] == "en":
            # Store tweet data in the 'training_tweets_collection' collection of the 'training_tweets' database, 
            # if the 'training_tweets_collection' does not already exist it will be created for you
            db.training_tweets_collection.insert_one(datajson)


if __name__ == "__main__":

    # these are provided to you through the Twitter API after you create a account
    consumer_key = "your_consumer_key"
    consumer_secret = "your_consumer_secret"
    access_token = "your_access_token"
    access_token_secret = 	"your_access_token_secret"

    auth1 = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth1.set_access_token(access_token, access_token_secret)

    '''LOCATIONS are the coordinate (long , lat) corners for a box that restricts the geographic area 
    from which you will receive tweets. The first two define the southwest corner of the box
    and the second two deine the northwest corner of the box. Below are the coordinates for 
    three boxes that define the contigous US, Alaska, and Hawaii'''
    LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,
                 -164.639405, 58.806859, -144.152365, 71.76871,
                 -160.161542, 18.776344, -154.641396, 22.878623]

    stream_listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True))
    stream = tweepy.Stream(auth=auth1, listener=stream_listener)
    stream.filter(locations=LOCATIONS)
