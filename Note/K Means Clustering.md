# K Means Clustering

* Create artificial data
* KMeans
* Compared with original data

## Create some Data


```python
from sklearn.datasets import make_blobs
```


```python
# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)
```


## Visualize Data


```python
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```
![](https://i.imgur.com/DEboFI4.png)


## Creating the Clusters


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(n_clusters=4)
```


```python
kmeans.fit(data[0])
```

```python
kmeans.cluster_centers_
```




    array([[-4.13591321,  7.95389851],
           [-9.46941837, -6.56081545],
           [-0.0123077 ,  2.13407664],
           [ 3.71749226,  7.01388735]])




```python
kmeans.labels_
```




    array([2, 3, 1, 3, 3, 0, 3, 1, 3, 1, 2, 1, 3, 3, 2, 1, 3, 1, 0, 2, 0, 1, 1,
           0, 2, 0, 0, 1, 3, 3, 2, 0, 3, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 2, 1, 3,
           2, 1, 0, 1, 1, 2, 3, 1, 0, 2, 1, 1, 2, 3, 0, 3, 0, 2, 3, 1, 0, 3, 3,
           0, 3, 1, 0, 1, 0, 3, 3, 1, 2, 1, 1, 0, 3, 0, 1, 1, 1, 2, 1, 0, 0, 0,
           0, 1, 1, 0, 3, 2, 0, 3, 1, 0, 1, 1, 3, 1, 0, 3, 0, 0, 3, 2, 2, 3, 0,
           3, 2, 2, 3, 2, 1, 2, 1, 2, 1, 3, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 3, 2,
           3, 1, 0, 3, 0, 2, 2, 3, 1, 0, 2, 2, 2, 2, 1, 3, 1, 2, 3, 3, 3, 1, 3,
           1, 1, 2, 0, 2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 3, 3, 0, 3, 2, 0, 0, 2,
           0, 0, 0, 0, 0, 1, 0, 3, 3, 2, 0, 1, 3, 3, 0, 1], dtype=int32)

```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```
![](https://i.imgur.com/Xk8pJOu.png)
