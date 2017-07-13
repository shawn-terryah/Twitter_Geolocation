from math import sin, cos, sqrt, atan2, radians
import pickle

def load_coord_dict():
    '''
    Input: n/a
    Output: A dictionary whose keys are the location names ('City, State') of the 
    379 classification labels and the values are the centroids for those locations
    (latitude, longitude)
    '''

    pkl_file = open("coord_dict.pkl", 'rb')
    coord_dict = pickle.load(pkl_file)
    pkl_file.close()
    return coord_dict

def compute_error_in_miles(zipped_predictions):
    '''
    INPUT: Tuple in the form of (predicted city, centroid of true location)
    OUTPUT: Float of the great-circle error distance between the predicted city
    and the true locaiton.
    '''
    
    radius = 3959   # approximate radius of earth in miles 
    
    predicted_city = zipped_predictions[0]
    actual_centroid = zipped_predictions[1]
    
    lat1 = radians(coord_dict[predicted_city][0])
    lon1 = radians(coord_dict[predicted_city][1])
    lat2 = radians(actual_centroid[0])
    lon2 = radians(actual_centroid[1])
    
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    
    a = sin(delta_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    error_distance = radius * c
    return error_distance
    
def correct_outside_the_us_errors(row):
    '''
    Helper function to correct the errors to 0.0 for the users that were correctly predicted. This 
    is especially important for users outside the US since they were all given the 
    same label ('NOT_IN_US, NONE') even though their centroids were all different.
    '''
    
    if row['predicted_location'] == row['geo_location']:
        error = 0.0
    else:
        error = row['error_in_miles']
    return error


if __name__ == "__main__":
    
    # Load coord_dict
    coord_dict = load_coord_dict()
    
    centroid_of_true_location = evaluation_df['centroid'].values
    zipped_predictions = zip(predictions, centroid_of_true_location)
    
    # Create a new column with the error value for each prediction
    evaluation_df['error_in_miles'] = map(lambda x: compute_error_in_miles(x), zipped_predictions)
    
    # Change the error of correct predictions to 0.0 miles 
    evaluation_df['error_in_miles'] = evaluation_df.apply(lambda x: 
                                                          correct_outside_the_us_errors(x),
                                                           axis=1)
                                                           
    median_error = evaluation_df['error_in_miles'].median()
