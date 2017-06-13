from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def plot_contiguous_US_tweets(lon, lat, file_path):
    '''
    INPUT: List of longitudes (lon), list of latitudes (lat), file 
    path to save the file (file_path) 
    OUTPUT: Plot of tweets in the contiguous US.
    '''

    map = Basemap(projection='merc',
                  resolution = 'h', area_thresh = 10000,
                  llcrnrlon=-140.25, llcrnrlat=5.0,
                  urcrnrlon=-56.25, urcrnrlat=54.75)

    x,y = map(lon, lat)
    map.plot(x, y, 'bo', markersize=2, alpha=.3)
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    map.fillcontinents(color = '#DAF7A6', lake_color='#a7cdf2')
    map.drawmapboundary(fill_color='#a7cdf2')

    plt.gcf().set_size_inches(15,15)
    plt.savefig(file_path, format='png', dpi=1000)
    
    
def plot_Alaska_tweets(lon, lat, file_path):
    '''
    INPUT: List of longitudes (lon), list of latitudes (lat), file 
    path to save the file (file_path) 
    OUTPUT: Plot of tweets in Alaska.
    '''

    map = Basemap(projection='merc',
                  resolution = 'h', area_thresh = 10000,
                  llcrnrlon=-174.25, llcrnrlat=53.0,
                  urcrnrlon=-135.25, urcrnrlat=72.75)

    x,y = map(lon, lat)
    map.plot(x, y, 'bo', markersize=2, alpha=.3)
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    map.fillcontinents(color = '#DAF7A6', lake_color='#a7cdf2')
    map.drawmapboundary(fill_color='#a7cdf2')

    plt.gcf().set_size_inches(3,3)
    plt.savefig(file_path, format='png', dpi=1000)
    
    
def plot_Hawaii_tweets(lon, lat, file_path):
    '''
    INPUT: List of longitudes (lon), list of latitudes (lat), file 
    path to save the file (file_path) 
    OUTPUT: Plot of tweets in Hawaii
    '''
    
    map = Basemap(projection='merc',
                  resolution = 'h', area_thresh = 10000,
                  llcrnrlon=-160.5, llcrnrlat=18.7,
                  urcrnrlon=-154.0, urcrnrlat=22.8)

    x,y = map(lon, lat)
    map.plot(x, y, 'bo', markersize=2, alpha=.3)
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    map.fillcontinents(color = '#DAF7A6', lake_color='#a7cdf2')
    map.drawmapboundary(fill_color='#a7cdf2')

    plt.gcf().set_size_inches(3,3)
    plt.savefig(file_path, format='png', dpi=1000)
