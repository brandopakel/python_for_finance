import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pylab as plt
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

rs = RandomState(MT19937(SeedSequence(123456789)))

X, y = make_blobs(n_samples=250, centers=4, random_state=500, cluster_std=1.25) # type: ignore # creates the sample data set for clustering with 250 samples and 4 centers

model = KMeans(n_clusters=4, random_state=0) # instantiates model object, given certain parameters; knowledge about the sample data is used to inform the instantation

model.fit(X=X) # fits the model object to the raw data

y_kmeans = model.predict(X) # predicts the cluster (number) given the raw data

# Gaussian model 
model = GaussianMixture(n_components=4, random_state=0)

model.fit(X=X)

y_gm = model.predict(X=X)

print(y_kmeans[:12]) # shows some cluster numbers as predicted
print((y_gm == y_kmeans).all()) # results from the k-means and gaussian mixture are the same

'''plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='coolwarm')
plt.show()'''