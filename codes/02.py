import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

def plot_feature(feature):
    fig, ax = plt.subplots()

    colors = ['blue', 'red', 'green']

    for i in range( len(iris.target_names) ):
        ax.hist(
                iris.data[iris.target==i, feature], 
                label=iris.target_names[i],
                color=colors[i]
                )

    ax.set_xlabel(iris.feature_names[feature])
    ax.legend(loc='upper right')
    plt.show()

plot_feature(0)
plot_feature(1)
plot_feature(2)
plot_feature(3)
