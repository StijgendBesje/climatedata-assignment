#%%

'''
add jupyter magic marker to show figures in visual code
'''
import pandas as pd
import numpy as np  
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from typing 


class Descriptives:
    '''
    following https://towardsdatascience.com/data-exploration-and-analysis-using-python-e564473d7607
    '''
    def __init__(self, data, data_dummy):
        self.data = data
        self.data_dummy = data_dummy
        self.figure_path = "/home/tosca/Projects/AssignmentIrisData/data/"

    def assess_outliers(self):
        '''
         data.select_dtypes(include=['float64']):
          two possible outliers:
        petal_length iris-versicolor
        sepal_length iris- virginica 
        '''
        filename = "boxplot_by_class.png"
        
        if os.path.isfile(f"{self.figure_path}boxplot/{filename}"):
            os.remove(f"{self.figure_path}boxplot/{filename}")

        boxplot = data.boxplot(by = 'class')
        plt.savefig(f"{self.figure_path}boxplot/{filename}")
        plt.close()

    def assess_distribution(self, column):
        '''
        Univariate Analysis, differs between categorical and continous
        normal distribution: sepal_length, width 
        Petal Length, width : two distributions 
        equal distrubtion for each class. 
        '''
        
        if os.path.isfile(f"{self.figure_path}hist/{column}"):
            os.remove(f"{self.figure_path}hist/{column}")

        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        plt.title(label = column)
        plt.hist(self.data[column])
        plt.savefig(f"{self.figure_path}hist/hist_{column}.png")
        plt.show()
        plt.close()

    def bivariate_analysis(self):
        ''' 
        three  clear groups based on petal_length and sepal_length: 
        virginica: longer petal and sepal 
        versicolor: average petal length, wide variation in sepal_length
        setosa: petal length small, large variation in sepal_length

        three clear group based on petal_width and sepal_width
        virginica: width petal, but seperated sepal width 
        versicolor: average petal width, but small sepal width
        setosa: thin petal_width but thick sepal_width

        seemed possible sepalwidth, length correlation over groups
        same for petal length and width correlation over groups
        '''
        return sns.relplot(x = 'sepal_length', y = 'petal_length', hue = 'class', data = self.data), \
            sns.relplot(x="sepal_width", y = "petal_width", hue="class", data = self.data), \
                sns.relplot(x="petal_width", y = "petal_length", hue="class", data = self.data), \

    def assess_correlation(self):
        '''
        Conclusion:  
        petal_length and sepal_length are strongly correlated (0,82)
        petal_width and sepal_length is strongly correlated(0,82)
        petal_width and petal_length are strongly correlated (0,96), indirect correlation?
        sepal_length and petal_length are strongly correlated (0,87)
        iris setosa and petal_length and width strongly negatively correlated
        '''

        if os.path.isfile(f"{self.figure_path}heatmap/corr_heatmap.png"):
            os.remove(f"{self.figure_path}heatmap/corr_heatmap.png")

        corr = self.data_dummy.corr()
        plt.figure(figsize=(10,10))
        ax = sns.heatmap(corr, annot = True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
        ax.set_yticklabels(ax.get_yticklabels(), rotation = 45)
        plt.savefig(f"{self.figure_path}heatmap/corr_heatmap.png")
        plt.close()

    def assess_covariation(self):
        sns.pairplot(self.data, hue = 'class', size = 3)
        sns.set()
# %%


if __name__ == '__main__':
    data = pd.read_csv("/home/tosca/Projects/AssignmentIrisData/data/iris.data")
    data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    Description = Descriptives(data, pd.get_dummies(data))
   
    for i in data.columns:
        Description.assess_distribution(i)

    Description.bivariate_analysis()
    Description.assess_outliers()
    Description.assess_correlation()
    Description.assess_covariation()
