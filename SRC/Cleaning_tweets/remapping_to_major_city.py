import numpy as np
import pickle

def load_US_coord_dict():
    '''
    Input: n/a
    Output: A dictionary whose keys are the location names ('City, State') of the 
    378 US classification locations and the values are the centroids for those locations
    (latitude, longittude)
    '''

    pkl_file = open("US_coord_dict.pkl", 'rb')
    US_coord_dict = pickle.load(pkl_file)
    pkl_file.close()
    return coord_dict
    
def find_dist_between(tup1, tup2):
    '''
    INPUT: Two tuples of latitude, longitude coordinates pairs for two cities
    OUTPUT: The distance between the cities
    '''
    
    return np.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

def closest_major_city(tup):
    '''
    INPUT: A tuple of the centroid coordinates for the tweet to remap to the closest major city
    OUTPUT: String, 'City, State', of the city in the dictionary 'coord_dict' that is closest to the input city 
    '''
    
    d={}
    for key, value in US_coord_dict.iteritems():
        dist = find_dist_between(tup, value)
        if key not in d:
            d[key] = dist
    return min(d, key=d.get)

def get_closest_major_city_for_US(row):
    '''
    Helper function to return the closest major city for US users only. For users
    outside the US it returns 'NOT_IN_US, NONE' 
    '''
    
    if row['geo_location'] == 'NOT_IN_US, NONE':
        return 'NOT_IN_US, NONE'
    else:
        return closest_major_city(row['centroid'])
        
        
if __name__ == "__main__":

    # Load US_coord_dict
    US_coord_dict = load_US_coord_dict()

    # Create a new column called 'closest_major_city'
    df['closest_major_city'] = df.apply(lambda row: get_closest_major_city_for_US(row), axis=1)
