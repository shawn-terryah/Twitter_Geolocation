



'''
Plot tweets in the contiguous US.
'''

# 'llcrnrlon' = lower left corner longitude
# 'llcrnrlat' = lower left corner latitude
# 'urcrnrlon' = upper right corner longitude
# 'urcrnrlat' = upper right corner latitude
# these define the bounding box of coordinates to plot
map = Basemap(projection='merc',
              resolution = 'h', area_thresh = 10000,
              llcrnrlon=-140.25, llcrnrlat=5.0,
              urcrnrlon=-56.25, urcrnrlat=54.75)

# 'lon' and 'lat' are lists of longitudes and latitudes
lon = longs
lat = lats
x,y = map(lon, lat)

map.plot(x, y, 'bo', markersize=2, alpha=.3)
map.drawcoastlines()
map.drawstates()
map.drawcountries()
map.fillcontinents(color = '#DAF7A6', lake_color='#a7cdf2')
map.drawmapboundary(fill_color='#a7cdf2')

plt.gcf().set_size_inches(15,15)
plt.show()
#plt.savefig('tweet_training_Contiguous.png', format='png', dpi=1000)
