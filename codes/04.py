import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

def plot_scatter_matrix():
    n = len(iris.feature_names)
    fig, ax = plt.subplots(n, n, figsize=(16, 16))

    colors = ['blue', 'red', 'green']

    for x in range(n):
        for y in range(n):
            for i in range( len(iris.target_names) ):
                ax[x, y].scatter(iris.data[iris.target==i, x], 
                                iris.data[iris.target==i, y],
                                label=iris.target_names[i],
                                c=colors[i])

            ax[x, y].set_xlabel(iris.feature_names[x])
            ax[x, y].set_ylabel(iris.feature_names[y])
            ax[x, y].legend(loc='upper left')
    
    plt.show()

plot_scatter_matrix()
plt.show()
