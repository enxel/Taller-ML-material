import random
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

wine = load_wine()

datasets = train_test_split(wine.data, 
                            wine.target,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets

p = Perceptron(random_state=42,
               max_iter=30,
               tol=0.001)
p.fit(train_data, train_labels)

sample = random.sample(range(len(train_data)), 10)
for i in sample:
    print(i, p.predict([train_data[i]]))

print(classification_report(p.predict(train_data), train_labels))

print(classification_report(p.predict(test_data), test_labels))
