# %% [markdown]
# Naive Bayes Classification

# %% [markdown]
# Importing The Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn as sk

# %% [markdown]
# importing the Dataset

# %%
df=pd.read_csv("Data.csv")
df

# %%
x=df.iloc[:,:-1].values
x

# %%
y=df.iloc[:,-1].values
y

# %% [markdown]
# Splitting The Data Into Training and Testing Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

# %%
x_train

# %%
x_test

# %%
y_train

# %%
y_test

# %% [markdown]
# feature selection

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

# %%
x_train

# %%
x_test

# %% [markdown]
# Fitting The Model On The Training Set

# %%
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

# %% [markdown]
# Prediction of a New Result

# %%
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Creating and Plotting Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True, fmt="g")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("Confussion Matrix.png")


# %% [markdown]
# Accuracy Score

# %%
print(accuracy_score(y_test,y_pred))

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_pred,y_test))


