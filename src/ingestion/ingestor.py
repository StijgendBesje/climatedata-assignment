
import numpy as np 
import pandas as pd
import os  

class Dataset:
   def __init__(self):
       self.source = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
       self.data = pd.read_csv("data/iris.data")
       self.data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
       

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
        
