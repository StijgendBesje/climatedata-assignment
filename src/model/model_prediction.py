#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from typing import List

from sklearn.utils import _determine_key_type



class DataModel:
    ''' 
    retraining model with hypertuning
    https://scikit-learn.org/stable/modules/grid_search.html
    '''
    def __init__(self, x_train, x_val, y_train, y_val, feature_list):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.feature_list = feature_list
    
    def establish_baseline(self):
        '''
        create baseline with historical averages 
        and calculate absolute error 
        '''
        baseline_preds = self.x_val[:, self.feature_list.index('average')]
        self.baseline_errors = abs(baseline_preds - self.y_val)
        print(f'average baseline error: {round(np.mean(self.baseline_errors),2)} degrees')

    def train_model(self):
        '''
        best accuracy of SVM
        '''
        self.rf = RandomForestRegressor(n_estimators= 1000, random_state=42)
        self.rf.fit(self.x_train, self.y_train)
        return self.rf, self.x_val
    
    def train_model_small(self):
        '''
        best accuracy of SVM
        '''
        self.rf_small = RandomForestRegressor(n_estimators= 1000, random_state=42, max_depth=3)
        self.rf_small.fit(self.x_train, self.y_train)

        return self.rf_small, self.x_val

    def train_model_important_feature(self):
        self.rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
        important_indices = [self.feature_list.index('temp_1'), self.feature_list.index('average')]
        train_important = self.x_train[:, important_indices]
        test_important = self.x_val[:, important_indices]

        self.rf_most_important.fit(train_important, self.y_train)

        return self.rf_most_important, test_important

    def make_predictions(self, func):
        rf,  x_val = func()
        self.predictions = rf.predict(x_val)
        self.errors = abs(self.predictions - self.y_val)

        print('Mean Absolute Error:', round(np.mean(self.errors), 2), 'degrees.')

        self.determine_metrics()
   
    def determine_metrics(self):
        mape = 100 * (self.errors/self.y_val)
        self.accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(self.accuracy, 2), '%.')

    def prediction_visualization(self):
        '''
        interpret model and report results
        https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
         '''
        pass


if __name__ == '__main__':
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split

    dirname = os.getcwd()
    data = pd.read_csv(f"{str(dirname)}/data/temps.csv")
    data_dummy = pd.get_dummies(data)

    label = np.array(data_dummy['actual'])
    
    df_features = data_dummy.drop('actual', axis = 1)
    features_list = list(df_features.columns)
    features = np.array(df_features)
     
    x_train, x_val, y_train, y_val = train_test_split( features, label, 
    test_size = 0.2, random_state=42)
    data_model = DataModel(x_train, x_val, y_train, y_val, features_list)

    data_model.establish_baseline()
    data_model.make_predictions(data_model.train_model)
    data_model.make_predictions(data_model.train_model_small)
    data_model.make_predictions(data_model.train_model_important_feature)

    
# %%
