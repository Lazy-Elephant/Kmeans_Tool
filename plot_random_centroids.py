import matplotlib.pyplot as plt
import numpy as np

"""This function will divvy up the full length of the vector into a series of centroid 
centered scatterplots so that a centroided data set can be made
length is the total number of points and centroids is the total number of"""
def random_data_generator(length, centroids):
    x = []
    y = []
    i = 1
    d = []
    #creates a list of random x and y values in clusters based on how many clusters are requested.
    while i <= centroids:
        center = (np.random.randint(-200000,200000), np.random.randint(-200000,200000))
        for n in range (int(length/centroids)):
            x_hold = center[0] + np.random.randint(-20,20)
            x.append(x_hold)
            y_hold = center[1] + np.random.randint(-20,20)
            y.append(y_hold)
        i += 1

    for j in range(len(x)):
        point = [x[j],y[j]]
        d.append(point)

    return d
data = random_data_generator(1000, 4)
print(len(data))
data = random_data_generator(1000, 14)
print(len(data))
data = random_data_generator(1000, 40)
print(len(data))