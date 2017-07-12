import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_error_histogram(df):
    '''
    Input: DataFrame that contains a 'error_in_miles' column
    Output: Histogram of errors
    '''
    
    plt.figure(figsize=(11,7))
    plt.hist(df['error_in_miles'], bins=20, normed=1)
    plt.xlabel('Error in miles', fontsize=20)
    plt.ylabel('Percentage', fontsize=22)
    plt.tick_params(labelsize=14)
    plt.show()
