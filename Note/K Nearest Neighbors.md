# K Nearest Neighbors


* Read Data
* Stabdardize the data to the same scale
* Train test split
* Plot out a variety of k values

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

## Get the Data

```python
df = pd.read_csv("Classified Data",index_col=0)
```


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913917</td>
      <td>1.162073</td>
      <td>0.567946</td>
      <td>0.755464</td>
      <td>0.780862</td>
      <td>0.352608</td>
      <td>0.759697</td>
      <td>0.643798</td>
      <td>0.879422</td>
      <td>1.231409</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.635632</td>
      <td>1.003722</td>
      <td>0.535342</td>
      <td>0.825645</td>
      <td>0.924109</td>
      <td>0.648450</td>
      <td>0.675334</td>
      <td>1.013546</td>
      <td>0.621552</td>
      <td>1.492702</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.721360</td>
      <td>1.201493</td>
      <td>0.921990</td>
      <td>0.855595</td>
      <td>1.526629</td>
      <td>0.720781</td>
      <td>1.626351</td>
      <td>1.154483</td>
      <td>0.957877</td>
      <td>1.285597</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.234204</td>
      <td>1.386726</td>
      <td>0.653046</td>
      <td>0.825624</td>
      <td>1.142504</td>
      <td>0.875128</td>
      <td>1.409708</td>
      <td>1.380003</td>
      <td>1.522692</td>
      <td>1.153093</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.279491</td>
      <td>0.949750</td>
      <td>0.627280</td>
      <td>0.668976</td>
      <td>1.232537</td>
      <td>0.703727</td>
      <td>1.115596</td>
      <td>0.646691</td>
      <td>1.463812</td>
      <td>1.419167</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

## Standardize the Variables

Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters. Any variables that are on a large scale will have a much larger effect on the distance between the observations, and hence on the KNN classifier, than variables that are on a small scale.

* #將資料常態分布化，平均值會變為0, 標準差變為1，使離群值影響降低
* MinMaxScaler與StandardScaler類似

```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()
```


```python
scaler.fit(df.drop('TARGET CLASS',axis=1))
```

```python
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
```


```python
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.123542</td>
      <td>0.185907</td>
      <td>-0.913431</td>
      <td>0.319629</td>
      <td>-1.033637</td>
      <td>-2.308375</td>
      <td>-0.798951</td>
      <td>-1.482368</td>
      <td>-0.949719</td>
      <td>-0.643314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.084836</td>
      <td>-0.430348</td>
      <td>-1.025313</td>
      <td>0.625388</td>
      <td>-0.444847</td>
      <td>-1.152706</td>
      <td>-1.129797</td>
      <td>-0.202240</td>
      <td>-1.828051</td>
      <td>0.636759</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.788702</td>
      <td>0.339318</td>
      <td>0.301511</td>
      <td>0.755873</td>
      <td>2.031693</td>
      <td>-0.870156</td>
      <td>2.599818</td>
      <td>0.285707</td>
      <td>-0.682494</td>
      <td>-0.377850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.982841</td>
      <td>1.060193</td>
      <td>-0.621399</td>
      <td>0.625299</td>
      <td>0.452820</td>
      <td>-0.267220</td>
      <td>1.750208</td>
      <td>1.066491</td>
      <td>1.241325</td>
      <td>-1.026987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.139275</td>
      <td>-0.640392</td>
      <td>-0.709819</td>
      <td>-0.057175</td>
      <td>0.822886</td>
      <td>-0.936773</td>
      <td>0.596782</td>
      <td>-1.472352</td>
      <td>1.040772</td>
      <td>0.276510</td>
    </tr>
  </tbody>
</table>
</div>

## Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)
```

## Using KNN

Remember that we are trying to come up with a model to predict whether someone will TARGET CLASS or not. We'll start with k=1.


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
knn.fit(X_train,y_train)
```


```python
pred = knn.predict(X_test)
```

## Predictions and Evaluations

Let's evaluate our KNN model!


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,pred))
```

    [[125  18]
     [ 13 144]]



```python
print(classification_report(y_test,pred))
```

                 precision    recall  f1-score   support
    
              0       0.91      0.87      0.89       143
              1       0.89      0.92      0.90       157
    
    avg / total       0.90      0.90      0.90       300
    

## Choosing a K Value

Let's go ahead and use the elbow method to pick a good K Value:


```python
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```


```python
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```