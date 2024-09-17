""" We trained six classifiers—Random Forest, Support Vector Machine (SVM), k-Nearest Neighbors (kNN), Multi-Layer Perceptron (MLP),
AdaBoost, and Gaussian Naive Bayes—using the top 20 most important voice features. These classifiers were used to calculate the 
average classification accuracy, which served as the utility score for the anonymized speech files, and were compared to the original speech files.
The input file, 'Input.csv', contains the acoustic features of the speech files (rows) and the column 'DX', which represents the dementia
diagnosis labels (NC, MCI, DE) for the classification task. The file 'Input_features_importance.csv' lists all acoustic features
in the 'Feature' column, sorted in descending order by importance. The 'num_important_features' column specifies the number of 
top-ranked features to use in the classification task. Finally, the classification accuracy for all six classifiers is computed
and printed using the 10-fold cross-validation method.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


precision = 4
label = ['DX'] #diagnosis label column
num_important_features = 20 #number of important voice features

df_label = pd.read_csv("Input.csv", sep=",", usecols=label)
df_cols = pd.read_csv("Input_features_importance.csv", sep=",")
features = df_cols.loc[0:num_important_features-1,'Feature']
df = pd.read_csv("Input.csv", sep=",", usecols=features)

cols = len(df.columns)
X = df
y = df_label

try:
            #RF classifier
            clf_RF = RandomForestClassifier(n_estimators=100)
            scores_RF = cross_val_score(clf_RF, X, y, scoring='accuracy', cv=10)
            accuracy_RF = round(scores_RF.mean(), precision)
            print("Random forest finished.")
except ValueError:
            print('Value error for RF classification!')    
try:
            #SVM classifier
            clf_SVM = SVC(kernel='rbf', C=1)
            scores_SVM = cross_val_score(clf_SVM, X, y, scoring='accuracy', cv=10)
            accuracy_SVM = round(scores_SVM.mean(), precision)
            print("SVM finished.")
except ValueError:
            print('Value error for SVM classification!')

try:
            #KNN classifier
            clf_KNN = KNeighborsClassifier(n_neighbors = 5)
            scores_KNN = cross_val_score(clf_KNN, X, y, scoring='accuracy', cv=10)
            accuracy_KNN = round(scores_KNN.mean(), precision)
            print("KNN finished.")
except ValueError:
            print('Value error for KNN classification!')  

try:
            #MLP classifier
            clf_MLP = MLPClassifier()
            scores_MLP = cross_val_score(clf_MLP, X, y, scoring='accuracy', cv=10)
            accuracy_MLP = round(scores_MLP.mean(), precision)
            print("MLP finished.")
except ValueError:
            print('Value error for MLP classification!')   

try:
            #Ada boost classifier
            clf_Ada = AdaBoostClassifier()
            scores_Ada = cross_val_score(clf_Ada, X, y, scoring='accuracy', cv=10)
            accuracy_Ada = round(scores_Ada.mean(), precision)
            print("Ada boost finished.")
except ValueError:
            print('Value error for Ada boost classification!')  

try:
            #GaussianNB classifier
            clf_GaussianNB = GaussianNB()
            scores_GaussianNB = cross_val_score(clf_GaussianNB, X, y, scoring='accuracy', cv=10)
            accuracy_GaussianNB = round(scores_GaussianNB.mean(), precision)
            print("GaussianNB finished.")
except ValueError:
            print('Value error for GaussianNB classification!')              

print('Random forest accuracy:',accuracy_RF)
print('SVM accuracy:',accuracy_SVM)
print('KNN accuracy:',accuracy_KNN)  
print('MLP accuracy:',accuracy_MLP)  
print('Ada boost accuracy:',accuracy_Ada)  
print('GaussianNB accuracy:',accuracy_GaussianNB)  