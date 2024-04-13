# %% [markdown]
# Kernel SVM

# %% [markdown]
# Importing The Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing The Data Set

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
# Splitting the data into Train and Test Set

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)
print(x_train,x_test,y_train,y_test)

# %% [markdown]
# Scaling the data set

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train,x_test)

# %% [markdown]
# Fitting the data set on the training set

# %%
from sklearn.svm import SVC
classifier=SVC(kernel="rbf",random_state=0)
classifier.fit(x_train,y_train)


# %% [markdown]
# Prediction of test set

# %%
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

# %% [markdown]
# Confussion Matrix

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
cm= confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True,fmt='g')
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title("Confussion Matrix")
plt.show()

# %% [markdown]
# Accuracy Score

# %%
print(accuracy_score(y_test,y_pred))

# %% [markdown]
# Classification Report

# %%
print(classification_report(y_test,y_pred))


