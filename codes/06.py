import numpy as np
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()

def distance(instance1, instance2):
    return np.linalg.norm(np.subtract(instance1, instance2))

def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
                  distance):

    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

data, labels = wine.data, wine.target

train_data, test_data, train_labels, test_labels = train_test_split(
                                                                    data, labels, 
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=20
                                                                    )

print("The first 7 data sets:")
print(test_data[:7])
print("The corresponding 7 labels:")
print(test_labels[:7])

for i in range(5):
    neighbors = get_neighbors(train_data, 
                              train_labels, 
                              test_data[i], 
                              3, 
                              distance=distance)
    print("Index:         ",i,'\n',
          "Testset Data:  ",test_data[i],'\n', 
          "Testset Label: ",test_labels[i],'\n', 
          "Neighbors:      ",neighbors,'\n',
          ", result of vote: ", vote(neighbors))

