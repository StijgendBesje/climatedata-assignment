#%%

'''
add jupyter magic marker to show figures in visual code
'''
import pandas as pd
import os
import matplotlib.pyplot as plt

class Descriptives:
    '''
    following https://towardsdatascience.com/data-exploration-and-analysis-using-python-e564473d7607
    '''
    def __init__(self, data):
        self.data = data
        self.dirname = os.getcwd()

    def assess_distribution(self, column):
        '''
        Univariate Analysis, differs between categorical and continous
        normal distribution: sepal_length, width 
        Petal Length, width : two distributions 
        equal distrubtion for each class. 
        '''
        
        if os.path.isfile(f"{self.dirname}/data/hist/{column}"):
            os.remove(f"{self.dirname}/data/hist/{column}")

        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.title(label = column)
        plt.hist(self.data[column])
        plt.savefig(f"{self.dirname}/data/hist/hist_{column}.png")
        plt.show()
        plt.close()

# %%


if __name__ == '__main__':
    from datetime import datetime

    dirname = os.getcwd()
    data = pd.read_csv(f"{str(dirname)}/data/temps.csv")

    # used to develop, no outliers removed
    Description = Descriptives(data, pd.get_dummies(data))
   
    for i in data.columns:
        Description.assess_distribution(i)

