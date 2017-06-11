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

        # Use us_tweets database
        db = client.us_tweets

        # Decode JSON
        datajson = json.loads(data)

        # I'm only storing tweets in English
        if "lang" in datajson and datajson["lang"] == "en":
            # Store tweet info into the us_tweets_collection collection.
            db.us_tweets_collection.insert_one(datajson)


if __name__ == "__main__":

    consumer_key = "7uxFa7rtaADBZn8sqB6mZBAQe"
    consumer_secret = 	"mBKqGXnZRHfq2yHzAjWBKQILb2d0bNvb2rNuINziLmOjmDdACz"
    access_token = "857859260335759360-DlIh1PZAHxjoXlkq2nFjgZ7Lvcs03A0"
    access_token_secret = 	"thQ7lro6ByXZyjnfXgGJcjMt26XT84adIMzSnC2dPdZD8"

    auth1 = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth1.set_access_token(access_token, access_token_secret)

    # api = tweepy.API(auth)

    '''LOCATIONS are the coordinate (long , lat) corners for a box that bins,
    where the twitter must come from. The first pair is the southwest
    corner and the second pair is the northwest corner of the box. The below
    pairs are for the contigous US, Alaska, and Hawaii'''
    LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,
                 -164.639405, 58.806859, -144.152365, 71.76871,
                 -160.161542, 18.776344, -154.641396, 22.878623]


    stream_listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True))
    stream = tweepy.Stream(auth=auth1, listener=stream_listener)
    stream.filter(locations=LOCATIONS)
