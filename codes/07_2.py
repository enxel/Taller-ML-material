from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

wine = load_wine()

data, labels = wine.data, wine.target

train_data, test_data, train_labels, test_labels = train_test_split(
                                                                    data, labels, 
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=20
                                                                    )

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels) 

print("Predictions from the classifier:")
learn_data_predicted = knn.predict(train_data)
print(learn_data_predicted)
print("Target values:")
print(train_labels)
print(accuracy_score(learn_data_predicted, train_labels))

print("Predictions from the classifier:")
test_data_predicted = knn.predict(test_data)
print(test_data_predicted)
print("Target values:")
print(test_labels)
print(accuracy_score(test_data_predicted, test_labels))

cm = confusion_matrix(test_data_predicted, test_labels)
print(cm) 

print(classification_report(knn.predict(train_data), train_labels))
