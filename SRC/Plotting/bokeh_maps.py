import pandas as pd
import pickle

from bokeh.plotting import figure, output_notebook, output_file, show
from bokeh.tile_providers import STAMEN_TERRAIN
output_notebook()

from functools import partial
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

# Web mercator bounding box for the US
US = ((-13884029, -7453304), (2698291, 6455972))

x_range, y_range = US
plot_width = int(900)
plot_height = int(plot_width*7.0/12)

def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height, **plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p


def plot_predictions_for_a_city(df, name_of_predictions_col, city):
    '''
    INPUT: DataFrame with location predictions; name of column in DataFrame that 
    contains the predictions; city ('City, State') to plot predictions for
    OUTPUT: Bokeh map that shows the actual location of all the users predicted to
    be in the selected city
    '''
    
    df_ = df[df[name_of_predictions_col] == city]
    
    # Initialize two lists to hold all the latitudes and longitudes
    all_lats = []
    all_longs = []
    
    # Pull all latitudes in 'centroid' column append to all_lats
    for i in df_['centroid']:
        all_lats.append(i[0])
    
    # Pull all longitudes in 'centroid' column append to all_longs
    for i in df_['centroid']:
        all_longs.append(i[1])
    
    # Initialize two lists to hold all the latitudes and longitudes 
    # converted to web mercator
    all_x = []
    all_y = []
    
    # Convert latittudes and longitudes to web mercator x and y format
    for i in xrange(len(all_lats)):
        pnt = transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(init='EPSG:3857')), 
                Point(all_longs[i], all_lats[i]))
        all_x.append(pnt.x)
        all_y.append(pnt.y)
    
    p = base_plot()
    p.add_tile(STAMEN_TERRAIN)
    p.circle(x=all_x, y=all_y, line_color=None, fill_color='#380474', size=15, alpha=.5)
    output_file("stamen_toner_plot.html")
    show(p)

if __name__ == "__main__":

    # Load pickled evaluation_df with location predictions
    evaluation_df_with_predictions = pd.read_pickle('evaluation_df_with_predictions.pkl')
    
    # Plot actual locations for users predicted to be in Eugene, OR
    plot_predictions_for_a_city(evaluation_df_with_predictions, 'predicted_location', 'Eugene, OR')
