from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydot
import os


class DataVisualization:
    def __init__(self, data_model):
        self.rf = data_model.rf
        self.rf_small = data_model.rf_small
        self.rf_most_important = data_model.rf_most_important
        self.feature_list  = data_model.feature_list
        self.dirname = os.getcwd()
    
    def visualize_tree_graph(self):
        tree = self.rf.estimators_[5]
        export_graphviz(tree, out_file = f'{self.dirname}/data/tree/tree.dot', feature_names = self.feature_list, rounded = True, precision = 1)
        (graph, ) = pydot.graph_from_dot_file(f'{self.dirname}/data/tree/tree.dot')
        graph.write_png(f'{self.dirname}/data/tree/tree.png')
        print('The depth of this tree is:', tree.tree_.max_depth)

    def visualize_tree_small(self):
        tree = self.rf_small.estimators_[5]
        export_graphviz(tree, out_file = f'{self.dirname}/data/tree/tree_small.dot', feature_names = self.feature_list, rounded = True, precision = 1)
        (graph, ) = pydot.graph_from_dot_file(f'{self.dirname}/data/tree/tree_small.dot')
        graph.write_png(f'{self.dirname}/data/tree/tree_small.png')
        print('The depth of this tree is:', tree.tree_.max_depth)

    def variable_importance(self):
        importances = list(self.rf.feature_importances_)
        feature_importances = [(feature, round(importance,2)) for feature, importance in zip(self.feature_list, importances)]
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        for feature, importance in feature_importances: 
            print(f'variable {feature} has importance : {importance}')

        return importances
    
    def variable_importance_visualiation(self):
        importances = self.variable_importance()
        x_values = list(range(len(importances)))
        plt.bar(x_values, importances, orientation = 'vertical')
        plt.xticks(x_values, self.feature_list, rotation= 'vertical')
        plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
        plt.savefig(f'{self.dirname}/data/tree/variable_importance.jpg')


