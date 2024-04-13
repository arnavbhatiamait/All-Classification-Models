# %% [markdown]
# Logistic Regression

# %% [markdown]
# Importing The Liberaries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# importing The Data Set

# %%

df=pd.read_csv("Data.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
y

# %% [markdown]
# Spliting Data set Into test and training sets

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# %%
x_train

# %%
x_test


# %%
y_train

# %%
y_test

# %% [markdown]
# Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# %%
x_test

# %%
x_train

# %% [markdown]
# Training The Logistic Regression model On Training Set

# %%
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# %% [markdown]
# Predicting The Test Set Results

# %%
y_pred=classifier.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# %% [markdown]
# Making The Confusion Matrix 

# %%
from sklearn.metrics import confusion_matrix,accuracy_score
cnf=confusion_matrix(y_test,y_pred)
print(cnf)
accuracy_score(y_test,y_pred)

# %% [markdown]
# Heatmap

# %%
import seaborn as sns
className=[0,1]
fig,ax=plt.subplots()
tick_marks=np.arange(len(className))
plt.xticks(tick_marks,className)
plt.yticks(tick_marks,className)
# heatmap
sns.heatmap(pd.DataFrame(cnf),annot=True,cmap="YlGnBu",fmt="g")
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title("Confusion Matrix",y=1.1)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
plt.savefig("Confussion Matrix.png")


# %% [markdown]
# 

# %% [markdown]
# Classification Report

# %%
from sklearn.metrics import classification_report
target_names = ['Buy' , 'Dont Buy']
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown]
# 


