from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
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


rnn = RadiusNeighborsClassifier(radius=30.0)
rnn.fit(train_data, train_labels) 

print("Predictions from the classifier:")
test_data_predicted = rnn.predict(test_data)
print(test_data_predicted)
print("Target values:")
print(test_labels)
results = [test_labels[i]==test_data_predicted[i] for i in range(len(test_labels)) ]
print(f"Overall results: {results}")
print(accuracy_score(test_data_predicted, test_labels))

print("Predictions from the classifier:")
learn_data_predicted = rnn.predict(train_data)
print(learn_data_predicted)
print("Target values:")
print(train_labels)
results = [train_labels[i]==learn_data_predicted[i] for i in range(len(train_labels)) ]
print(f"Overall results: {results}")
print(accuracy_score(learn_data_predicted, train_labels))

cm = confusion_matrix(test_data_predicted, test_labels)
print(cm) 

print(classification_report(rnn.predict(train_data), train_labels))
