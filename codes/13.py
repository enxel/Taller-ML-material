import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dataset = pd.read_csv('zoo.csv')
dataset = dataset.drop('animal_name',axis=1)

train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]

dectr = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

prediction = dectr.predict(test_features)

print("The prediction accuracy is: ", dectr.score(test_features,test_targets)*100,"%")

tree.plot_tree(decision_tree=dectr)

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
tree.plot_tree(clf)

plt.show()
