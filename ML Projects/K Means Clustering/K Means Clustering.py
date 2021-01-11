import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('College_Data',index_col=0)

df.head()
df.info()
df.describe()

## EDA
sns.set_style('whitegrid')

# ** Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# **Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)

# ** Create a stacked histogram showing Out of State Tuition based on the Private column. Try doing this using [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist'). **
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)

# **Create a similar histogram for the Grad.Rate column.**
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?**
df[df['Grad.Rate'] > 100]

df['Grad.Rate']['Cazenovia College'] = 100
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

## K Means Cluster Creation
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private',axis=1))

# ** What are the cluster center vectors?**
kmeans.cluster_centers_

## Evaluation
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))