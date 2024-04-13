# %% [markdown]
# Support Vector Machine (SVM)

# %% [markdown]
# Importing Libraries
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing Data Set

# %%
df=pd.read_csv("Data.csv")
df
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x,y)

# %% [markdown]
# splitting The Data Set Into training and testing Sets
# 

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)
print(x_train,x_test,y_train,y_test)

# %% [markdown]
# Features Scalling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)
print(x_train,x_test)

# %% [markdown]
# Traninig the SVM model on the training set

# %%
from sklearn.svm import SVC
classifier=SVC(kernel="linear",random_state=0)
classifier.fit(x_train,y_train)

# %% [markdown]
# Prediction of X_test with the model

# %%
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Making The confusion matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
cm= confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True,fmt="g")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion matrix")
plt.show()
plt.savefig("ConfusionMatrix.png")

# %% [markdown]
# Accuracy Score

# %%
accuracy_score(y_test,y_pred)

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_test,y_pred))


