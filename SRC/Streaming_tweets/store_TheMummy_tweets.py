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

        # Use Mongo the_mummy database
        db = client.the_mummy

        # Decode JSON
        datajson = json.loads(data)

        # I'm only storing tweets in English
        if "lang" in datajson and datajson["lang"] == "en":
            # Store tweet info into the the_mummy_collection of the_mummy database.
            db.the_mummy_collection.insert_one(datajson)


if __name__ == "__main__":

    consumer_key = 
    consumer_secret = 
    access_token = 
    access_token_secret = 

    auth1 = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth1.set_access_token(access_token, access_token_secret)

    # The twitter API will return tweets that match these phrases, regardlesss of order or case
    TRACK = ['TheMummy', '#TheMummy', '@themummy', 'The Mummy']

    stream_listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True))
    stream = tweepy.Stream(auth=auth1, listener=stream_listener)
    stream.filter(track=TRACK)
