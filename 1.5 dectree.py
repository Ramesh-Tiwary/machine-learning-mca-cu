from datetime import date
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load dataset
iris = load_iris("Iris.csv")
 
#  convert to DataFrame
df = pd.DataFframe(iris.data, columns=iris.feature_names)
df['ID'] = iris.target

#features selection and train.test split
x= date.drop(columns=['ID'])
y= date['ID']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
# calculating gini index(GI)
def gini_index(ID):
    classes, counts = np.unique(ID, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities ** 2)
    return gini

print("Gini Index of the target variable :", gini_index(y))

def information_gain(parent, left_child, right_child):
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)
    gain = gini_index(parent) - (weight_left * gini_index(left_child) + weight_right * gini_index(right_child))
    return gain

#example split (petal length feature)

threshold = x['petal length (cm)'].median()
left_child = y[x['petal length (cm)'] <= threshold]
right_child = y[x['petal length (cm)'] > threshold]
gain = information_gain(y, left_child, right_child)
print("Information Gain for petal length split at threshold", threshold, ":", gain)
# Create Decision Tree Classifier



