import seaborn as sns
iris = sns.load_dataset('iris')

import pandas as pd
import matplotlib.pyplot as plt

# Setosa is the most separable. 
sns.pairplot(iris,hue='species',palette='Dark2')

# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = iris[iris['species']=='setosa']
sns.kdeplot(x='sepal_width', y='sepal_length', data=setosa,
                 cmap="plasma", shade=True, shade_lowest=False)

# Train Test Split
from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a Model
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

# Model Evaluation
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Gridsearch Practice
from sklearn.model_selection import GridSearchCV

# Create a dictionary called param_grid and fill out some parameters for C and gamma.
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))