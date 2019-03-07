import numpy
import pandas as pd
from sklearn.metrics import accuracy_score as ac
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#csv open
df=pd.read_csv('hcc-data.csv')

#select rows and columns by number
X=df.iloc[:,5:31]
Y=df.iloc[:,32]

#Split arrays or matrices into random train and test subsets
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.9)

#Classifiers
KNN=KNeighborsClassifier(n_neighbors=2)
TreeClassifier=DecisionTreeClassifier()

#Fitting the Data
KNN.fit(x_train,y_train)
TreeClassifier.fit(x_train,y_train)

#Predicting and Plotting the Data
KN_Prediction=KNN.predict(x_test)
Tree_Prediction=TreeClassifier.predict(x_test)

plt.title('Classification using  K-Nearest Neighbour')
plt.ylabel('STATES: 0: DIES 1: LIVES')
plt.xlabel('Ferritin Values')
plt.scatter(x_test.iloc[:,-4],KN_Prediction,marker="*",color="blue")
plt.show()

plt.title('Classification using Decision Tree')
plt.ylabel('STATES: 0: DIES 1: LIVES')
plt.xlabel('Ferritin Values')
plt.scatter(x_test.iloc[:,-4],Tree_Prediction,marker="*",color="blue")
plt.show()