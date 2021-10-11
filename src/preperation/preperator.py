import pandas as pd
from scipy.sparse import data
from sklearn.model_selection import train_test_split

class DataPrep:
    def __init__(self, df):
        self.df = df
    
    def create_dummy_variable(self):
        self.df_dummy = pd.get_dummies(self.df)     
        return self.df #, self.df_dummy


    def create_train_test_set(self):
        '''
        Returns x_train, x_validation, y_train, y_validation
        '''
        array = self.df.values
        label = array[:, -1]
        features = array[:, 0:4] #all rows, with only first four columns 
        validation_size = 0.20
        seed = 7
        return train_test_split( features, label, 
         test_size = validation_size, 
         random_state=seed)

if __name__== '__main__':
    df = pd.read_csv("/home/tosca/Projects/AssignmentIrisData/data/iris.data")
    data_object = DataPrep(df)
    data_object.create_dummy_variable()
    data_object.create_train_test_set()



        


