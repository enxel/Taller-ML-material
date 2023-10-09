import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.target)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

print(iris_df)

iris_df.loc[iris_df['target']==0,iris.feature_names[0]].hist(label=iris.target_names[0])
iris_df.loc[iris_df['target']==1,iris.feature_names[0]].hist(label=iris.target_names[1])
iris_df.loc[iris_df['target']==2,iris.feature_names[0]].hist(label=iris.target_names[2])

plt.grid(False)
plt.legend()
plt.xlabel(iris.feature_names[0])
plt.ylabel('Frequency')

plt.show()

pd.plotting.scatter_matrix(iris_df.iloc[:,0:2], 
                           c=iris.target, 
                           figsize=(16, 16)
                          )
plt.show()

pd.plotting.scatter_matrix(iris_df.iloc[:,0:4], 
                           c=iris.target, 
                           figsize=(16, 16)
                          )
plt.show()
