#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 13:40:48 2023

@author: anush
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.optimize as opt
import numpy as np 


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# Read the dataset for Cancer
folder_path = os.path.dirname(__file__)
file_path = 'cancerData.csv'
df_cancer = pd.read_csv(os.path.join(folder_path, file_path))


print(df_cancer.columns)

x = df_cancer['radius_mean']
y = df_cancer['texture_mean']

sns.kdeplot(x=df_cancer['radius_worst'],y=df_cancer['texture_worst'],
            hue='diagnosis',data=df_cancer);


# ---- Clustering Code Starts ----
kmeans = cluster.KMeans(n_clusters=3)
df_cancer['cluster'] = kmeans.fit_predict(df_cancer[['radius_mean', 
                                                 'texture_mean']])
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
df_cancer['cen_x'] = df_cancer.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
df_cancer['cen_y'] = df_cancer.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})

c = ['#DF2020', '#81DF20', '#2095DF']
df_cancer['c'] = df_cancer.cluster.map({0:c[0], 1:c[1], 2:c[2]})
fig, ax = plt.subplots()
ax.scatter(x, y, 
            c=df_cancer.c, alpha = 0.6, s=10)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(df_cancer['cen_x'], df_cancer['cen_y'], 10, "purple", marker="d")

z = np.polyfit(x, y, 2)
p = np.poly1d(z)

ax.plot(x, p(x), label='Fit', linewidth=0.2)

# ---- Clustering Code Ends ----



plt.show()

df_cancer['diagnosis'].value_counts().plot.pie(
    autopct="%.2f%%", colors=["#FA8128", "red"])




pd.plotting.scatter_matrix(df_cancer[['radius_mean',
                                      'texture_mean',
                                      'perimeter_mean',
                                      'texture_worst',
                                      'radius_worst']], 
                            figsize=(8, 8), s=5, alpha=0.6)













