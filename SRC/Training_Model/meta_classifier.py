import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
meta_y = meta_df['closest_major_city'].values

# Fit Random Forest with 'meta_X' and 'meta_y'
meta_RF = RandomForestClassifier(n_estimators=60, n_jobs=-1)
meta_RF.fit(meta_X, meta_y)

# Pickle the meta Random Forest for future use
joblib.dump(meta_RF, 'META_RF.pkl') 
