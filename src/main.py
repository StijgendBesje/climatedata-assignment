from matplotlib.pyplot import draw_if_interactive
from scipy.sparse import data
from ingestion import Dataset
from preperation import DataPrep
from description import Descriptives
from model import DataModel
from visualization import DataVisualization
import os

'''
rewrite the main
'''

dirname = os.getcwd()

def start_ingestion():
    data_class = Dataset()
    print(data_class.describe_data())
    data_class.find_na()

    return data_class.data

def start_descriptive_analytics(df):
    data_descr = Descriptives(df)
    for i in df.columns:
        data_descr.assess_distribution(i)

def start_parsing(df):
   data_prep =  DataPrep(df)
   data_prep.detecting_anomolies("scatterplot.jpg")
   data_prep.remove_anomolies()
   data_prep.detecting_anomolies("scatterplot_removed_anomolies.jpg")
   return data_prep.create_train_test_set()
   
def start_model(x_train, x_val, y_train, y_val, feature_list): 
    data_model = DataModel(x_train,x_val, y_train, y_val, feature_list)
    data_model.establish_baseline()
    data_model.make_predictions(data_model.train_model)
    data_model.make_predictions(data_model.train_model_small)
    data_model.make_predictions(data_model.train_model_important_feature)
    return data_model

def start_visualiation(data_model, df):
    data_visualization = DataVisualization(data_model)
    data_visualization.visualize_tree_graph()
    data_visualization.visualize_tree_small()
    data_visualization.variable_importance_visualiation()


    
if __name__ == '__main__':
    df = start_ingestion()
    start_descriptive_analytics(df)
    x_train, x_val, y_train, y_val, feature_list = start_parsing(df)
    data_model = start_model(x_train, x_val, y_train, y_val, feature_list)
    start_visualiation(data_model)
