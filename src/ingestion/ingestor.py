import numpy as np
import pandas as pd
import os  

dirname = os.getcwd()

class Dataset:
    '''
    Variables are: 
     year: year of observation
     month: number of month of the year
     day: number for day of the year
     week: day of the week as char string
     temp_2: max temperature two days prior
     temp_1: max temperature one day prior
     average: historical average max temperature 
     actual: max tempetature measurement
     friend: friends prediction, +/- 20 from average
    '''
    
    def __init__(self):
       self.source = f"{str(dirname)}/data/temps.csv"
       self.data = pd.read_csv(self.source, header = 0)

    def describe_data(self):
       '''
       Describe the content of the data, general information and the data types 
       '''
       return self.data.describe(), self.data.info(), self.data.dtypes
  
    def find_na(self): 
        if self.data.isnull().values.any():
            raise ValueError("NA values need to be cleaned")
        else: 
            print("all data correct")
        

if __name__ == "__main__": 
    data_object = Dataset()
    data_object.describe_data()
    data_object.find_na()