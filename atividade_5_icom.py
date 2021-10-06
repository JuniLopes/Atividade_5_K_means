"""
Created on Thu Sep 27 17:54:19 2021

# Equipe:
# *   Juliane Bezerra
# *   Rubens Lopes
# *   Aline Soares

"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

#importing the Iris dataset with pandas
parkinson = pd.read_csv('D:/OneDrive/Faculdade/S7/parkinson/parkinson_formated.csv')

data = parkinson

kmeans = KMeans(n_clusters = 2, init = 'random')

kmeans.fit(data)

kmeans.cluster_centers_

distance = kmeans.fit_transform(data)

labels = kmeans.labels_

data = data.iloc[:,:-1].values

plt.scatter(data[:,0], data[:,-1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 100, c = 'red',label = 'Centroids')
plt.title('Parkinson Clusters and Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend('Parkinson')
plt.show()