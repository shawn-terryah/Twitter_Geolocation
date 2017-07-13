from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def plot_contiguous_US_tweets(lon, lat, file_path):
    '''
    INPUT: List of longitudes (lon), list of latitudes (lat), file path to save the plot (file_path) 
    OUTPUT: Plot of tweets in the contiguous US.
    '''
    
    map = Basemap(projection='merc',
                  resolution = 'h', 
                  area_thresh = 10000,
                  llcrnrlon=-140.25,    # lower left corner longitude of contiguous US
                  llcrnrlat=5.0,        # lower left corner latitude of contiguous US
                  urcrnrlon=-56.25,     # upper right corner longitude of contiguous US
                  urcrnrlat=54.75)      # upper right corner latitude of contiguous US

    x,y = map(lon, lat)

    map.plot(x, y, 'bo', markersize=2, alpha=.3)
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    map.fillcontinents(color = '#DAF7A6', lake_color='#a7cdf2')
    map.drawmapboundary(fill_color='#a7cdf2')
    plt.gcf().set_size_inches(15,15)
    plt.savefig(file_path, format='png', dpi=1000)
