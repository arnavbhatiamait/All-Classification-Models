# %% [markdown]
# K nearest Neighbours

# %% [markdown]
# Importing Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing Data Set

# %%
df=pd.read_csv("Data.csv")
df

# %%
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x,y)


# %% [markdown]
# splitting The Data Set Into training and testing Sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

# %%
print(x_train,x_test,y_train,y_test)

# %% [markdown]
# Features Scalling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_test=sc.fit_transform(x_test)
x_train=sc.fit_transform(x_train)

# %%
print(x_train,x_test)

# %% [markdown]
# Training the model on the training set

# %%
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
classifier.fit(x_train,y_train)

# %%
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape((len(y_pred)),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Metrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# %% [markdown]
# Plotting The Confussion Matrix 

# %%
import seaborn as sns
sns.heatmap(cm,fmt="g",annot=True)
plt.ylabel("Prediction")
plt.xlabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("ConfussionMatrix.png")

# %% [markdown]
# Classifiction Scores 

# %%
print(classification_report(y_test,y_pred))

# %% [markdown]
# Testing For Different Values of k

# %%
for i in range(1,20):
    classi=KNeighborsClassifier(n_neighbors=i,p=2,metric="minkowski")
    classi.fit(x_train,y_train)
    y_pred=classi.predict(x_test)
    score=accuracy_score(y_pred,y_test)
    print("K value = ",i)
    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_pred))
    print("Classification Report")
    print(classification_report(y_test,y_pred))
    print("Accuracy Score= ",score)
    


