import json
import unicodecsv as csv

def Tweets_JSON_to_CSV(file_list, csv_output_file):
    '''
    INPUT: list of JSON files
    OUTPUT: single CSV file

    Takes a list of JSON files containing tweets and associated metadata and reads each file
    line by line, parsing the revelent fields, and writing it to a CSV file.
    '''

    count = 0
    f = csv.writer(open(csv_output_file, "wb+"))
    f.writerow(['tweet', 'geo_country_code', "geo_state", 'geo_location', 'bounding_box', 'screen_name',
                'user_description', 'favourites_count', 'followers_count', 'statuses_count', 'friends_count',
                'listed_count', 'user_described_location', 'time_created_at', 'time_zone', 'utc_offset'])

    for file_ in file_list:
        with open(file_, "r") as r:
            for line in r:
                try:
                    tweet = json.loads(line)
                except:
                    continue
                if tweet and tweet['place'] != None:
                    f.writerow([tweet['text'], # Tweet
                                tweet['place']['country_code'], # Geolocated country code
                                tweet['place']['full_name'],  # Geolocated City, State
                                tweet['place']['full_name'], # Geolocated City, State
                                tweet['place']['bounding_box'], # Polygon coordinates
                                tweet['user']['screen_name'], # User's screen name
                                tweet['user']['description'], # User's description field
                                tweet['user']['favourites_count'], # User's favorites count
                                tweet['user']['followers_count'], # User's followers count
                                tweet['user']['statuses_count'], # Total number of user's tweets
                                tweet['user']['friends_count'], # User's friends count
                                tweet['user']['listed_count'], # User's listed count
                                tweet['user']['location'], # User provided location
                                tweet['created_at'], # Time tweet was sent
                                tweet['user']['time_zone'], # User provided time zone
                                tweet['user']['utc_offset']]) # UTC offset
                    count += 1
                    if count % 100000 == 0:
                        print 'Just stored tweet #{}'.format(count)
