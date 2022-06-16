import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from plot_random_centroids import random_data_generator

#gets the kmeans from a dataset based on the number
#of clusters selected
def get_kmeans (training_data, clusters):
    kmeans = sk.cluster.KMeans(n_clusters=clusters, random_state=0).fit(training_data)
    return kmeans

# creates a list of kmeans to be later used for comparing cluster numbers
def kmean_list_for_comparison(training_data, median_cluster, delta):
    kmeans = []
    i = delta
    mc = median_cluster
    #section will go half of delta below and half of delta above if it can, and if not will go from 1 to delta
    if mc > int(i/2) :
        mc = int(mc-i/2)
        for n in range(i):
            kmeans.append (get_kmeans(training_data, mc+n))
    else:
        n = 1
        end = i
        while n < end:
            kmeans.append(get_kmeans(training_data,n))
            n+=1
    return kmeans

#takes in training data and a delta value of how many neighboring clusters to check
#returns information used to select the optimal cluster number
def test_clusters(kmeans, data, delta, median_cluster):
    #creates the x axis which is the number of centroids
    if median_cluster > int(delta/2) :
        start = median_cluster-int(delta/2)
        end = int(delta/2)+median_cluster
    else:
        start = 1
        end = median_cluster + (delta-median_cluster)
    base = np.linspace(start,end, delta)
    var = []
    #takes in the list of kmeans with varied centroid values, predicts where a point from data goes,
    #then calculates the squared distance of the point from its centroid as a measure of variation
    n=0
    for elem in kmeans:
        var.append(0)
        for x in data:
            x = [x]
            cent_index = elem.predict(x)
            cent = elem.cluster_centers_[cent_index]
            var[n] += np.sqrt(np.abs(x[0][0]-cent[0][0])**2 + np.abs(x[0][1]-cent[0][1])**2)
        var[n] /= len(data)
        n+=1
    #elbow plot
    # plt.subplot(2,2,1)
    # plt.plot(base, var)
    # plt.xlabel('centroids')
    # plt.ylabel('variation')
    # plt.show()
    return base,var


"""This is the main calling function of this .py file Call this function with a rough approximation for the
number of centroids that the data has, a training dataset, and a testing dataset in the form of list of 2 member
lists as shown to the right: [[x,y],[x,y]]. Delta is how many centroid values will be checked, delta/2 to the left 
and delta/2 to the right from the approximate centroid value selected by the user. returns the x and y of the 
 elbow curve"""

"""Generally the optimal number of centroids is at the value where the variation slope levels off"""
def kmeans_elbow_plot(centroids, delta, training_data, testing_data):
    x = []
    y = []
    for n in testing_data:
        x.append(n[0])
        y.append(n[1])
    # plt.subplot(2,2,2)
    # plt.scatter(x,y, s=1)
    kmeans = kmean_list_for_comparison(training_data, centroids, delta)
    x,y = test_clusters(kmeans, testing_data, delta, centroids)
    return x,y


# centroids = 40
# delta = 10
# data = random_data_generator(1000,centroids)
# kmeans_elbow_plot(centroids, delta, data, data)

