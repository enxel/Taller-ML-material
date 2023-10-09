import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

def scatter_features(feature1, feature2):
    fig, ax = plt.subplots()

    colors = ['blue', 'red', 'green']

    for i in range( len(iris.target_names) ):
        ax.scatter(
                    iris.data[iris.target==i, feature1], 
                    iris.data[iris.target==i, feature2],
                    label=iris.target_names[i],
                    c=colors[i]
                    )

    ax.set_xlabel(iris.feature_names[feature1])
    ax.set_ylabel(iris.feature_names[feature2])
    ax.legend(loc='upper left')
    plt.show()


for i in range(3):
    for j in range(3):
        if not( i == j):
            scatter_features(i,j)
