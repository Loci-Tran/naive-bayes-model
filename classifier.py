"""
Author: Loc Dinh Quang Tran
Student ID: 111080421
Course name: 11120BAI700500 Introduction of Statistics and Machine Learning (II)
Teacher: Dr. 李政霖 (John)
Homework 01 problem 01
References:
https://python.plainenglish.io/naive-bayes-classification-algorithm-in-practice-40dd58d18df4

"""
import numpy as np 
import pandas as pd 	


import numpy as np 
import pandas as pd 	

class  NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, C):
        """
        Input:
            X (DataFrame, N samples, D features): Training data truncating labels.
            C (DataFrame, N samples, classes): following labels with training data X
        Output:
            None

        The function computes the likelihoods of X when classified on the class C
        self.likelihoods is the dictionary type, which is stored the likelihoods of each feature.
        self.priors is the DataFrame type, which is stored the prior probability of each class.
        E.g.
        >> clf.likelihoods
        {'Weight Loss': {'Obvious': {'Yes': 0.625, 'No': 0.167}, 
                         'Mild': {'Yes': 0.25, 'No': 0.5}, 
                         'No': {'Yes': 0.125, 'No': 0.333}}, 
        'Headache':     {'No': {'Yes': 0.625, 'No': 0.834}, 
                         'Yes': {'Yes': 0.375, 'No': 0.167}}, 
        'Fever':        {'Yes': {'Yes': 0.625, 'No': 0.333}, 
                         'No': {'Yes': 0.375, 'No': 0.667}}, 
        'Cough':        {'Yes': {'Yes': 0.375, 'No': 0.667}, 
                         'No': {'Yes': 0.625, 'No': 0.333}}}
        P('Weight Loss' = 'Obvious' | C = 'Yes') = 0.625
        P('Weight Loss' = 'Obvious' | C = 'No') = 0.625
        
        >> cls.priors
            Classes
        Yes  0.571429
        No   0.428571
        P(C = 'Yes') = 0.571429, P(C = 'No') = 0.428571
        """
        C.rename(columns={str(C.columns[0]) : 'Classes'}, inplace=True)     # Rename the class column as Classes
        classes = C['Classes'].value_counts().to_frame()                    # Count number of occurrence of class variable
        classes['Classes'] = (classes['Classes'] / classes['Classes'].sum())# Compute the prior probability
        class_vars = classes.index.to_list()                                # Extract class variables

        self.priors = classes
        self.class_vars = class_vars
        self.features = X.columns

        likelihoods = {}                                                    
        for feature in self.features:                                       # Iterate feature
            feature_var = X[feature].value_counts().index.to_list()         # Extract feature variables

            feature_var_dict = {}                                           # return e.g. {'Obvious': {'Yes': 0.625, 'No': 0.167}, 'Mild': {'Yes': 0.25, 'No': 0.5}, 'No': {'Yes': 0.125, 'No': 0.333}}
            for var in feature_var:                                         # Iterate variables in feature
                var_rows = X[X[feature] == var]                             # Collect variable rows

                var_dict = {}                                               # return e.g. {'Yes': 0.625, 'No': 0.167}
                for i in class_vars:                                        # Iterate variable in class
                    i_rows = C.loc[C['Classes'] == i]                       # Collect class variable rows (all Yes or all No)
                    var_i_rows = var_rows.index.intersection(i_rows.index)  # Intersect row indeces between val_rows and i_rows = row(feature variable, class variable)
                    var_dict[i] = len(var_i_rows) / len(i_rows)             # Compute likelihood: P(feature variable | class = class variable)
                
                feature_var_dict[var] = var_dict                            # Embed var_dict into feature_var_dict
            likelihoods[feature] = feature_var_dict                         # Embed feature_var_dict into likelihoods
        self.likelihoods = likelihoods
        return self

    def predict(self, X_test):
        """
        Input:
            X_test (DataFrame, N samples, D features): tested data. 
            Number of feature must be the same as self.features
        Output:
            predict_results (list): The predicted classifying X_test into classes
            predict_prob (list): The highest posterior probability of X_test
        
        E.g. Classify two vector X_test (indeces 0, 1)
        >> X_test = pd.DataFrame({'Weight Loss': ['Obvious', 'Mild'], 'Headache': ['Yes', 'No'], 'Fever': ['No', 'No'], 'Cough': ['No', 'Yes']})
          Weight Loss Headache Fever Cough
        0     Obvious      Yes    No    No
        1        Mild       No    No   Yes
        
        >> predict_results, predict_prob = clf.predict(X_test)
        predict_results = ['Yes', 'No']
        predict_prob = [0.92227, 0.8634]
        P[X0 | C = 'Yes'] = 0.9223
        P[X1 | C = 'No'] = 0.8634
        """
        predict_results = []
        predict_prob = []
        for i in range(len(X_test)):                                        # Iterate each X_test vector 0, 1, 2,...
            results = {}                                                    # Store P(X | C=i) * P(C = i)     
            for j in self.class_vars:                                       # Iterate each class variable Yes, No
                results[j] = float(self.priors.loc[j])                      # P(C = j)

                likelihoods_X = 1                                           # P(Xi | Cj); i -> feature; j -> class variable
                for feature in self.features:                               # Iterate each feature
                    likelihoods_X *= self.likelihoods[feature][X_test[feature][i]][j] # P(X | Cj) = ∏ P(Xi | Cj) -> Product of likelihoods
                results[j] *= likelihoods_X                                 # P(C = j) * P(X | Cj) 

            results_sum_i = sum(results.values())                           # ∑ P(C = j) * P(X | Cj)
            for i in results.keys():
                results[i] = results[i] / results_sum_i                     # P(C = j) * P(X | Cj) / sum = Posterior probability
            
            predict_i = max(results, key=results.get)                       # Classify vector Xi into classes
            predict_results.append(predict_i)                               # Append it to predict_results
            predict_prob.append(round(results[predict_i], 4))               # Append its probability to predict_prob
        
        return predict_results, predict_prob

if __name__ == "__main__":
    path = "./data.xlsx"
    df = pd.read_excel(path, header=0)

    X_test = pd.DataFrame({'Weight Loss': ['Obvious', 'Mild'], 'Headache': ['Yes', 'No'], 'Fever': ['No', 'No'], 'Cough': ['No', 'Yes']})
    X_train = df.drop(['Prescription'], axis=1) # Array of training vector Xi
    y_train = df['Prescription'].to_frame() # Array of training class Ci corresponding to Xi

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))

    
