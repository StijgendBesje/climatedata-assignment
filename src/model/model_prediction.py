#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



class DataModel:
    def __init__(self, x_train, x_val, y_train, y_val):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.models = []

    def predication_models(self):
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))

        self.results = []
        self.names = []
        for name, model in self.models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, self.x_train, self.y_train, cv=kfold, scoring='accuracy')
            self.results.append(cv_results)
            self.names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    def prediction_visualization(self):
        plt.boxplot(self.results, labels=self.names)
        plt.title('Algorithm Comparison')
        plt.show()

    def make_predictions(self):
        '''
        best accuracy of SVM
        '''
        model = SVC(gamma='auto')
        model.fit(self.x_val, self.y_val)
        self.predictions = model.predict(self.x_val)


    def test_harnass(self):
        '''
        A Gentle Introduction to k-fold Cross-Validation
        Introduction to Random Number Generators for Machine Learning in Python
        '''
        print(f"accuracy score is {accuracy_score(self.y_val, self.predictions)}")
        print(f"confusion score is \n {confusion_matrix(self.y_val, self.predictions)}")
        print(f"classification report is {classification_report(self.y_val, self.predictions)}")


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("/home/tosca/Projects/AssignmentIrisData/data/iris.data")
    
    array = df.values
    label = array[:, -1]
    features = array[:, 0:4] #all rows, with only first four columns 

    x_train, x_val, y_train, y_val = train_test_split( features, label, 
    test_size = 0.2, random_state=7)
    data_model = DataModel(x_train, x_val, y_train, y_val)

    data_model.predication_models()
    data_model.prediction_visualization()
    data_model.make_predictions()
    data_model.test_harnass()

# %%
