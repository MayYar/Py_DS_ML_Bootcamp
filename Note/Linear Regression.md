# Linear Regression

### Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


## Training a Linear Regression Model

We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

### X and y arrays


```python
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
```

## Train Test Split

split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```

## Creating and Training the Model


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
# do not set this equal to any other variable object
lm.fit(X_train,y_train)
```

## Model Evaluation

evaluate the model by checking out it's coefficients and how we can interpret them.


```python
# print the intercept
print(lm.intercept_)
```

    -2640159.79685



```python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Area Income</th>
      <td>21.528276</td>
    </tr>
    <tr>
      <th>Avg. Area House Age</th>
      <td>164883.282027</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Rooms</th>
      <td>122368.678027</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Bedrooms</th>
      <td>2233.801864</td>
    </tr>
    <tr>
      <th>Area Population</th>
      <td>15.150420</td>
    </tr>
  </tbody>
</table>
</div>

Interpreting the coefficients:

- Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.

## Predictions from our Model


```python
predictions = lm.predict(X_test)
```


```python
plt.scatter(y_test,predictions)
```
![](https://i.imgur.com/rXjPkht.png)

**Residual Histogram**


```python
sns.distplot((y_test-predictions),bins=50);
```

![](https://i.imgur.com/ClJUQAF.png)

* If you have normally distributed resuduals, it means your model was a correct choice for the data

## Regression Evaluation Metrics


Here are three common evaluation metrics for regression problems:

**Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:

$$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

**Mean Squared Error** (MSE) is the mean of the squared errors:

$$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

**Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:

$$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

Comparing these metrics:

- **MAE** is the easiest to understand, because it's the average error.
- **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
- **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

All of these are **loss functions**, because we want to minimize them.

```python
from sklearn import metrics
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 82288.2225191
    MSE: 10460958907.2
    RMSE: 102278.829223