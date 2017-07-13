import json
import unicodecsv as csv    # unicodecsv ensures that emojis are preserved

def tweets_json_to_csv(file_list, csv_output_file):
    '''
    INPUT: list of JSON files
    OUTPUT: single CSV file
    
    Takes a list of JSON files containing tweets and associated metadata and reads each file
    line by line, parsing the revelent fields, and writing it to a CSV file.
    '''

    count = 0
    f = csv.writer(open(csv_output_file, "wb+"))
    
    # Column names
    f.writerow(['tweet',                    # relabelled: the API calls this 'text'
                'country_code', 
                'geo_location',             # relabelled: the API calls this 'full_name'
                'bounding_box', 
                'screen_name',
                'favourites_count', 
                'followers_count', 
                'statuses_count', 
                'friends_count',
                'listed_count', 
                'user_described_location',  # relabelled: the API calls this 'location'
                'created_at', 
                'utc_offset'])

    for file_ in file_list:
        with open(file_, "r") as r:
            for line in r:
                try:
                    tweet = json.loads(line)
                except:
                    continue
                if tweet and tweet['place'] != None:
                    f.writerow([tweet['text'],                                    
                                tweet['place']['country_code'],                   
                                tweet['place']['full_name'],                      
                                tweet['place']['bounding_box']['coordinates'],    
                                tweet['user']['screen_name'],                    
                                tweet['user']['favourites_count'],                
                                tweet['user']['followers_count'],                 
                                tweet['user']['statuses_count'],                  
                                tweet['user']['friends_count'],                 
                                tweet['user']['listed_count'],          
                                tweet['user']['location'],                      
                                tweet['created_at'],                             
                                tweet['user']['utc_offset']])                   
                    count += 1
                    
                    # Status update
                    if count % 100000 == 0:
                        print 'Just stored tweet #{}'.format(count)
                        
if __name__ == "__main__":
    
    tweets_json_to_csv(['training_tweets.json'], 'training_tweets.csv')
