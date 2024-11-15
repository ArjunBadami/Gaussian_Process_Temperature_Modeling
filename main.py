#UCSD WINTER 2024
#CSE 251U
#Homework 7.8

#Name: Arjun H. Badami
#PID: A13230476

import json
from collections import defaultdict

import sklearn
from sklearn import linear_model
import random
import gzip
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.manifold import MDS
from numpy.linalg import norm
from scipy.cluster import hierarchy
import pandas

gptest = np.genfromtxt('gptest.csv', delimiter=',', dtype=None)
#pandas.read_csv('gptest.csv', header=None)

gptrain = np.genfromtxt('gptrain.csv', delimiter=',', dtype=None)

def calc_l2_distance(a, b):
    if(len(a) != len(b)):
        return -1
    ss = 0
    for i in range(len(a)):
        ss += (a[i] - b[i]) ** 2

    return (ss**0.5)

def calc_covar(a, b):
    return math.exp(-1*(calc_l2_distance(a,b))/1.5)

def calc_mse(a, b):
    if(len(a) != len(b)):
        return -1
    ss = 0
    for i in range(len(a)):
        ss += (a[i] - b[i]) ** 2

    return ss/len(a)

X_tr = [[row[0], row[1]] for row in gptrain]
X_te = [[row[0], row[1]] for row in gptest]

y_tr = [row[2] for row in gptrain]
y_te = [row[2] for row in gptest]

K_tr = [[1]*len(y_tr) for _ in range(len(y_tr))]
K_te = [[1]*len(y_te) for _ in range(len(y_te))]

K = [[1]*(len(y_tr)+len(y_te)) for _ in range((len(y_tr)+len(y_te)))]

for i in range((len(y_tr)+len(y_te))):
    for j in range(i):
        point1 = X_tr[0]
        point2 = X_tr[0]
        if(i >= len(y_tr)):
            point1 = X_te[i-len(X_tr)]
        else:
            point1 = X_tr[i]

        if (j >= len(y_tr)):
            point2 = X_te[j - len(X_tr)]
        else:
            point2 = X_tr[j]
        cov = calc_covar(point1, point2)
        K[i][j] = cov
        K[j][i] = cov


for i in range(len(y_tr)):
    for j in range(i):
        cov = calc_covar(X_tr[i], X_tr[j])
        K_tr[i][j] = cov
        K_tr[j][i] = cov


for i in range(len(y_te)):
    for j in range(i):
        cov = calc_covar(X_te[i], X_te[j])
        K_te[i][j] = cov
        K_te[j][i] = cov


K_trte = np.array(K)[:len(y_tr), len(y_tr):]
K_tetr = np.array(K)[len(y_tr):, :len(y_tr)]

K_te = np.array(K_te)
K_tr = np.array(K_tr)
X_te = np.array(X_te)
X_tr = np.array(X_tr)

avg_y_tr = sum(y_tr)/len(y_tr)

m_te = np.array([20]*len(y_te))
m_tr = np.array([20]*len(y_tr))

N_mean = 20+(K_tetr @ inv(K_tr) @ (y_tr - m_tr))
N_var = K_te - (K_tetr @ inv(K_tr) @ K_trte)


MSE1 = calc_mse(y_te, N_mean)
MSE2 = calc_mse(y_te, [avg_y_tr]*len(y_te))
print('MSE1 = ' + str(MSE1))
print('MSE2 = ' + str(MSE2))



####PART C
min_lat = min(min(X_tr[:,0]), min(X_te[:,0]))
max_lat = max(max(X_tr[:,0]), max(X_te[:,0]))
min_long = min(min(X_tr[:,1]), min(X_te[:,1]))
max_long = max(max(X_tr[:,1]), max(X_te[:,1]))

d = 71
lat_lc = (max_lat - min_lat - 1) / 71
long_lc = (max_long - min_long - 1) / 71

lats = []
longs = []
currlat = min_lat + 0.5
currlong = min_long + 0.5
P5K = []

for i in range(d):
    lats.append(currlat)
    longs.append(currlong)

    currlat += lat_lc
    currlong += long_lc

for i in lats:
    for j in longs:
        P5K.append([i, j])

K_5000 = [[1]*(len(y_tr)+len(P5K)) for _ in range((len(y_tr)+len(P5K)))]

for i in range((len(y_tr)+len(P5K))):
    for j in range(i):
        point1 = X_tr[0]
        point2 = X_tr[0]
        if(i >= len(y_tr)):
            point1 = P5K[i-len(X_tr)]
        else:
            point1 = X_tr[i]

        if (j >= len(y_tr)):
            point2 = P5K[j-len(X_tr)]
        else:
            point2 = X_tr[j]
        cov = calc_covar(point1, point2)
        K_5000[i][j] = cov
        K_5000[j][i] = cov


K_5000_trte = np.array(K_5000)[:len(y_tr), len(y_tr):]
K_5000_tetr = np.array(K_5000)[len(y_tr):, :len(y_tr)]
K_5000_tr = np.array(K_5000)[:len(y_tr), :len(y_tr)]
K_5000_te = np.array(K_5000)[len(y_tr):, len(y_tr):]

N_mean_5000 = 20+(K_5000_tetr @ inv(K_5000_tr) @ (y_tr - m_tr))
N_var_5000 = K_5000_te - (K_5000_tetr @ inv(K_5000_tr) @ K_5000_trte)

print(len(N_mean_5000))

P5K = np.array(P5K)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(P5K[:,0], P5K[:,1], c=N_mean_5000, cmap='viridis', marker='o')
cbar = plt.colorbar(scatter, label='Temperatures')

# Set axis labels and title
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Predicted Temperatures')

plt.show()


NV5K = []
for i in range(len(N_mean_5000)):
    NV5K.append(N_var_5000[i][i])


plt.figure(figsize=(10, 8))
scatter = plt.scatter(P5K[:,0], P5K[:,1], c=NV5K, cmap='viridis', marker='o')
cbar = plt.colorbar(scatter, label='Standard Deviations')
scatter = plt.scatter(X_tr[:,0], X_tr[:,1], color='purple')
# Set axis labels and title
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Standard Deviations')

plt.show()