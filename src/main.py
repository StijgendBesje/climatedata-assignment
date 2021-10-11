from scipy.sparse import data
from ingestion import Dataset
from preperation import DataPrep
from description import Descriptives
from model import DataModel

'''
Ingest, parse, descriptive analytics of Iris dataset 

Hypothesis, when is a flower a setosa, versicolor or verginica 
what are key predictories 

K-means cluster is unsupervised, the variable is labeled so 
random forest 
'''

def start_ingestion():
    DataClass = Dataset()
    print(DataClass.describe_data())
    DataClass.find_na()

    return DataClass.data

def start_parsing(df):
   DataPreperation =  DataPrep(df)
   return DataPreperation.create_dummy_variable(), DataPreperation

def start_descriptive_analytics(df):
    DataDescription = Descriptives(df)
    print(DataDescription.assess_distribution())

def start_model(data_prep): 
    x_train, x_validation, y_train, y_validation = data_prep.create_train_test_set()
    data_model = DataModel(x_train,x_validation, y_train, y_validation)
    data_model.predication_models()
    data_model.prediction_visualization()
    data_model.make_predictions()
    data_model.test_harnass()
    
if __name__ == '__main__':
    df = start_ingestion()
    df, data_prep = start_parsing(df)
    start_model(data_prep)