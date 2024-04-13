# %% [markdown]
# Decission Trees Classification

# %% [markdown]
# Importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing the Dataset

# %%
df=pd.read_csv("Data.csv")
df.head()

# %%
x=df.iloc[:,:-1].values
x

# %%
y=df.iloc[:,-1].values
y

# %% [markdown]
# Train Test Split

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

# %%
x_test

# %%
x_train

# %%
y_train

# %%
y_test

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %%
x_train

# %%
x_test

# %% [markdown]
# Model training on the training data Set

# %%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(x_train,y_train)

# %% [markdown]
# Prediction of test set results

# %%
y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)

# %% [markdown]
# Plotting the confusion matrix

# %%
import seaborn as sns
sns.heatmap(cm,annot=True,fmt='g')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
plt.savefig("confusion_matrix.png")


# %% [markdown]
# Accuracy Score

# %%
print(accuracy_score(y_test,y_pred))

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_pred,y_test))


