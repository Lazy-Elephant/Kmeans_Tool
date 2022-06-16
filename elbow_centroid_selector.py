import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from plot_random_centroids import random_data_generator
from kmeans_elbow_plot import kmeans_elbow_plot


def data_splitter(data, fraction_train):
    train_data = []
    num_train = int(len(data)*fraction_train)
    i = 0
    while i <= num_train:
        n = np.random.randint(0,len(data)-1)
        train_data.append(data[n])
        del data[n]
        i+=1
    test_data = data
    return train_data, test_data

def optimal_k(x,y):
    n=1
    der = []
    while n < len(y):
        dy = y[n]-y[n-1]
        dx = x[n] - x[n-1]
        der.append(dy/dx)
        n += 1
    plt.plot(x[:len(x)-1],der)
    plt.show()
    for pnt in der:
        if np.abs(pnt) <= 0.5:
            opt_slope = pnt
            opt_index = der.index(pnt)
    return x[opt_index]
delta = 10
centroids = 40
length = 1000
data = random_data_generator(length, centroids)
training_data, testing_data = data_splitter(data, 1/4)
x,y = kmeans_elbow_plot(centroids, delta, training_data, testing_data)
optimal_k(x,y)
