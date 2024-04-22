# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5. Print the outputs
6. End the program.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SUDHIR KUMAR .R
RegisterNumber: 212223230221
*/
```
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
#Load data from CSV
data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data
#Extract features
X = data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
#Number of clusters
k = 5
#Initialize KMeans
kmeans = KMeans(n_clusters=k)
#Fit the data
kmeans.fit(X)
centroids = kmeans.cluster_centers_
#Get the cluster labels for each data point
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m'] #Define colors for each cluster
for i in range(k):
  cluster_points=X[labels==i] #Get data points belonging to cluster i
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
              color=colors[i],label=f'Cluster(i+1)')
  #Find minimum enclosing circle
distances=euclidean_distances(cluster_points,[centroids[i]])
radius=np.max(distances)
circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
plt.gca().add_patch(circle)
#Plotting the centroids
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.grid(True)
plt.axis('equal') #Ensure aspect ratio is equal
plt.show()
```

## Output:

### Dataset

![322936196-d1d27e0e-e107-42fd-aaa6-0fb847cf24dd](https://github.com/Sudhirr5/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139332214/7268c43c-4a31-41a5-a59a-6ee6d1c31c1d)

### Centroids and Labels

![322936377-986d0fdf-a82e-4265-a377-991b8c87e4d8](https://github.com/Sudhirr5/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139332214/999d0231-1d91-49f0-8eea-9d18040a744c)

### Graph

![322936524-213ce817-ab83-4a4c-8bcb-075d0cd95f37](https://github.com/Sudhirr5/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139332214/63510180-16d0-4dc7-a45b-2a8dc23e92ab)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
