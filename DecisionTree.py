import pandas as pd
import numpy as np
from pprint import pprint
import math
import sys

from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import _tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from six import StringIO  
from IPython.display import Image  


# Classification model for detecting breast cancer using a binary decision tree. 

# loading and preprocessing the data

with open('breast-cancer-wisconsin.data', 'r') as f:
    data_list = [l.strip('\n').split(',') for l in f if '?' not in l]


data_array = np.array(data_list).astype(int)   # training data
data = pd.DataFrame(data_list)
data = data.apply(pd.to_numeric)

# feature selection

x_train = data.iloc[:,[2,1,6,8,7,3]].values   # get features and labels for training data
x_test = data.iloc[:,10].values
print('feature selection done.')

# creating a Binary Decision Stump (decision tree depth = 1)
tree = DecisionTreeClassifier(random_state=12345, criterion='entropy')
model = tree.fit(x_train,x_test)

# visulaise decision tree

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data, filled = True, rounded = True, special_characters = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('binary_tree_entropy.png')
Image(graph.create_png())


#extracting rules from the tree

tree_rules = export_text(model, decimals=0)
print(tree_rules)
with open('tree.txt', 'w') as file:
   file.write(tree_rules)
   

# test data prediction

with open('test.txt') as f:
    test_data_list = [l.strip('\n').split(',') for l in f if '?' not in l]

test_array = np.array(test_data_list).astype(int)   # test data
test = pd.DataFrame(test_data_list)
test = test.apply(pd.to_numeric)

y_train = test.iloc[:,[3,2,7,9,8,4]].values

