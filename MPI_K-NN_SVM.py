#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import seaborn as sb

import time
from mpi4py import MPI

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
s = StandardScaler()


# In[34]:


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
cores = multiprocessing.cpu_count()
print('Cores in the computer:', cores)


# In[35]:



# Se carga el conjunto de datos

df0 = pd.read_csv("C:/Maestria/DS/0.csv", header=None )
df1 = pd.read_csv("C:/Maestria/DS/1.csv", header=None )
df = pd.concat([df1,df0], axis = 0)
#df


# In[45]:


# División de los datos en train y test

x = df.loc[:,0:63]
y = df[64]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=30)


# In[46]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[47]:


X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
X_train_s.shape, X_test_s.shape


# In[48]:


start_time = time.perf_counter()
if rank==0:
    
    hyperparameters = []
    for metric in ['minkowski','manhattan']:
        for p in range(1,3):
            knn = KNeighborsClassifier(n_neighbors=2, metric = metric, p=p)
            knn.fit(X_train_s,y_train)
            knn.fit(X_train_s, y_train)


            print('Accuracy of K-NN classifier on training set: {:.2f}'
                 .format(knn.score(X_train_s, y_train)))
            print('Accuracy of K-NN classifier on test set: {:.2f}'
                 .format(knn.score(X_test_s, y_test)))

            pred = knn.predict(X_test_s)
            print(confusion_matrix(y_test, pred))
            print(metric)
            print(p)
            print(classification_report(y_test, pred))

    
if rank==1:

    clf = SVC(kernel='rfb', gamma=0.01, C=100,decision_function_shape='ovr',probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc=accuracy_score(y_pred, y_test)
    print(classification_report(y_test,y_pred))

                
finish_time = time.perf_counter()

print(f"Program in rank {rank} finished in {finish_time-start_time} seconds")


# In[50]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm #Import svm model

#Create a svm Classifier and hyper parameter tuning 
ml = svm.SVC()
  
# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000], 
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(ml, param_grid,n_jobs=-1, refit = True, verbose = 1,cv=15)
start = time.time()
# fitting the model for grid search
grid_search=grid.fit(X_train_s,y_train)
end = time.time()
print(end - start)


# In[51]:


print(grid_search.best_params_)


# In[60]:


start_time = time.perf_counter()
clf= SVC(kernel='rbf',C=1000,gamma=0.01,decision_function_shape='ovr',probability=True)
clf.fit(X_train_s,y_train)
y_pred_svm=clf.predict(X_test_s)
#print(classification_report(y_test,y_pred_svm))

finish_time = time.perf_counter()
print(f"Reporte de Clasificación {classification_report(y_test,y_pred_svm)} finished in {finish_time-start_time} seconds")


# In[53]:


start_time = time.perf_counter()

if rank==0:
  
    for kernel in ['rbf','poly']:
        for gamma in ['auto','scale']:
            for C in range(1,2):

                clf = SVC(kernel=kernel, gamma=gamma, C=C)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc=accuracy_score(y_pred, y_test)
                print(kernel)
                print(gamma)
                print(C)
                print(classification_report(y_test,y_pred))

if rank==1:
    
    hyperparameters = []
    for metric in ['minkowski','manhattan']:
        for p in range(1,3):
            knn = KNeighborsClassifier(n_neighbors=2, metric = metric, p=p)
            knn.fit(X_train_s,y_train)
            knn.fit(X_train_s, y_train)


            print('Accuracy of K-NN classifier on training set: {:.2f}'
                 .format(knn.score(X_train_s, y_train)))
            print('Accuracy of K-NN classifier on test set: {:.2f}'
                 .format(knn.score(X_test_s, y_test)))

            pred = knn.predict(X_test_s)
            print(confusion_matrix(y_test, pred))
            print(metric)
            print(p)
            print(classification_report(y_test, pred))

                
finish_time = time.perf_counter()

print(f"Program in rank {rank} finished in {finish_time-start_time} seconds")


# In[56]:


pred = knn.predict(X_test_s)
f,ax = plt.subplots(figsize=(6, 6))
confusion_mtx = confusion_matrix(y_test, pred)
sns.set(font_scale=1.5)
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray",ax=ax)
plt.xlabel("Predicción")
plt.ylabel("Actual")
plt.title("Confusion Matrix Validation set")
plt.show()


# In[ ]:


clf = SVC(kernel='linear', gamma=0.01, C=15,decision_function_shape='ovr',probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc=accuracy_score(y_pred, y_test)
print(classification_report(y_test,y_pred))

clf = SVC(kernel='poly', gamma=0.01, C=15,decision_function_shape='ovr',probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc=accuracy_score(y_pred, y_test)
print(classification_report(y_test,y_pred))

clf = SVC(kernel='rbf', gamma=0.01, C=15,decision_function_shape='ovr',probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc=accuracy_score(y_pred, y_test)
print(classification_report(y_test,y_pred))


# In[ ]:




