# Support Vector Machine

* Sikit learn built in breast cancer dataset
* SVC model
* Use GridSezrch to find the best parameter

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Get the Data

```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer = load_breast_cancer()
```

The data set is presented in a dictionary form:


```python
cancer.keys()
```
    dict_keys(['DESCR', 'target', 'data', 'target_names', 'feature_names'])

## Set up DataFrame


```python
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
```

```python
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
```

## Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
# ravel()：將多維數組轉換成一維數組
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
```

```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#       decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#       max_iter=-1, probability=False, random_state=None, shrinking=True,
#       tol=0.001, verbose=False)
```



## Predictions and Evaluations

Now let's predict using the trained model.


```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
# [[  0  66]
#  [  0 105]]
```

    



```python
print(classification_report(y_test,predictions))
#                     precision    recall  f1-score   support
    
#               0       0.00      0.00      0.00        66
#               1       0.61      1.00      0.76       105
    
#     avg / total       0.38      0.61      0.47       171
```

                 
    

:warning: WARNING: 
```
/Users/marci/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
```
    
Notice that we are classifying everything into a single class! This means our model needs to have it parameters adjusted (it may also help to normalize the data).

We can search for parameters using a **GridSearch**!

# Gridsearch

Finding the right parameters (like what C or gamma values to use) is a tricky task! But luckily, we can be a little lazy and just try a bunch of combinations and see what works best! This idea of creating a 'grid' of parameters and just trying out all the possible combinations is called a Gridsearch, this method is common enough that Scikit-learn has this functionality built in with GridSearchCV! The CV stands for cross-validation which is the GridSearchCV takes a dictionary that describes the parameters that should be tried and a model to train. The grid of parameters is defined as a dictionary, where the keys are the parameters and the values are the settings to be tested. 


```python
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
```

> ***C*** controls the cost of misclassification on the  training data
> A large C value gives you low bias and high variance. Low bias because you penalize the cost of misclassification a lot for a larger C value. With a smaller C value you are not going to penalize that cost as much. So it gives you a higher bias and lower variance.

> ***gamma*** parameter has to do with the free parameter of the Gaussian radial basis function
> A small gamma means a Gaussian for large variance. And a large gamma value is going to lead to a high bias and low variance in model, implying that support vector does not have a widespread influence.

```python
from sklearn.model_selection import GridSearchCV
```

One of the great things about GridSearchCV is that it is a meta-estimator. It takes an estimator like SVC, and creates a new estimator, that behaves exactly the same - in this case, like a classifier. You should add refit=True and choose verbose to whatever number you want, higher the number, the more verbose (verbose just means the text output describing the process).


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
```

What fit does is a bit more involved then usual. First, it runs the same loop with cross-validation, to find the best parameter combination. Once it has the best combination, it runs fit again on all data passed to fit (without cross-validation), to built a single new model using the best parameter setting.


```python
# May take awhile!
grid.fit(X_train,y_train)
```

You can inspect the best parameters found by GridSearchCV in the best_params_ attribute, and the best estimator in the best\_estimator_ attribute:


```python
grid.best_params_
#{'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
```




    




```python
grid.best_estimator_
```

Then you can re-run predictions on this grid object just like you would with a normal model.


```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test,grid_predictions))

# [[ 60   6]
#  [  3 102]]
```

    



```python
print(classification_report(y_test,grid_predictions))

#                     precision    recall  f1-score   support
    
#               0       0.95      0.91      0.93        66
#               1       0.94      0.97      0.96       105
    
#     avg / total       0.95      0.95      0.95       171
```

                 