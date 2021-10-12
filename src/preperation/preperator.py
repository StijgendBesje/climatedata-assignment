#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np


class DataPrep:
    ''' 
    handling missing data: 
    https://machinelearningmastery.com/handle-missing-data-python/
    '''
    def __init__(self, data):
        self.data = data
        self.dirname = os.getcwd()
    
    def convert_to_datetime(self):
        years = self.data['year']
        months = self.data['month']
        days = self.data['day']

        dates = [str(int(year)) + '-' + str(int(month)) + '-' + \
            str(int(day)) for year, month, day in zip(years, months, days)]

        dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        return dates

    def create_dummy_variable(self):
        self.data_dummy = pd.get_dummies(self.data)     
        return self.data_dummy

    def detecting_anomolies(self, filename):
        '''
        Read Later
        https://www.dataquest.io/blog/making-538-plots/
        http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
        '''

        dates = self.convert_to_datetime()
        plt.style.use("fivethirtyeight")
        
        #Set up the plotting layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
        fig.autofmt_xdate(rotation = 45)

        # Actual max temperature measurement
        ax1.plot(dates, self.data['actual'])
        ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

        # Temperature from 1 day ago
        ax2.plot(dates, self.data['temp_1'])
        ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

        # Temperature from 2 days ago
        ax3.plot(dates, self.data['temp_2'])
        ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

        # Friend Estimate
        ax4.plot(dates, self.data['friend'])
        ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

        plt.tight_layout(pad=2)
        
        if os.path.isfile(f"{self.dirname}/data/anomoly_detection/{filename}"):
            os.remove(f"{self.dirname}/data/anomoly_detection/{filename}")

        plt.savefig(f"{self.dirname}/data/anomoly_detection/{filename}")
        plt.show()
        plt.close()

    def remove_anomolies(self):
        '''
        There is not enough data, therefore the anomoly is changed to the average in the same month
        '''
        anomoly_temp_1 = {}
        anomoly_avg_1 = self.data.groupby('month')['temp_1'].mean()
        month_1 = []
        anomoly_temp_2 = {}
        anomoly_avg_2 = self.data.groupby('month')['temp_2'].mean()
        month_2 = []

        for index, row in self.data.iterrows():
            if row['temp_1'] > 100:
                anomoly_temp_1[index] = row['temp_1']
                month_1.append(row['month'])
        
            if row['temp_2']> 100:
                anomoly_temp_2[index] = row['temp_2']
                month_2.append(row['month'])

        for i, month in zip(anomoly_temp_1, month_1):
            self.data.at[i, 'temp_1'] = anomoly_avg_1[month]
        
        for i, month in zip(anomoly_temp_2, month_2):
            self.data.at[i, 'temp_2'] = anomoly_avg_2[month]
        

    def create_features_target(self):
        self.create_dummy_variable()
        label = np.array(self.data_dummy['actual'])
        df_features = self.data_dummy.drop('actual', axis = 1)
        self.features_list = list(df_features.columns)
        features = np.array(df_features)
        
        return label, features, 
                
    def create_train_test_set(self):
        '''
        Returns x_train, x_validation, y_train, y_validation
        '''
        label, features = self.create_features_target()
        validation_size = 0.20
        seed = 7 # 42 is same reproducible results
        x_train, x_val, y_train, y_val =  train_test_split( features, label, test_size = validation_size, random_state=seed)

        if x_train.shape[0]  != y_train.shape[0] or x_val.shape[0] !=y_val.shape[0]:
            raise Exception("no equal validation and testing groups in length")
        if x_train.shape[1] != x_val.shape[1]:
            raise Exception("no equal amount of columns")

        return x_train, x_val, y_train, y_val, self.features_list

if __name__== '__main__':
    dirname = os.getcwd()

    df = pd.read_csv(f"{str(dirname)}/data/temps.csv")
    data_object = DataPrep(df)
    # preparing the data for descriptives
    data_object.detecting_anomolies("scatterplot.jpg")
    data_object.remove_anomolies()
    data_object.detecting_anomolies("scatterplot_removed_anomolies.jpg")
    # preparing the data for modelling
    print(data_object.create_train_test_set())
    



        


