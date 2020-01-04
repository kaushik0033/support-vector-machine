# --------------
# Loading the Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df=pd.read_csv(path)

# Check the correlation between each feature and check for null values
df.isna().sum()
corr_mat=df.corr()
df.label.value_counts()
# Print total no of labels also print number of Male and Female labels
mapper={"male":0,"female":1}
df['labal_']=df.label.apply(lambda x: mapper[x])
df.drop('label',axis=1,inplace=True)
corr_upper=corr_mat.where(np.triu(np.ones(corr_mat.shape),k=1).astype(np.bool))
drop_cols=[col for col in corr_upper if any(corr_upper[col]>0.95)]
# Label Encode target variable
df.drop(drop_cols,axis=1,inplace=True)
scalar=StandardScaler()
scalar_df=scalar.fit_transform(df.iloc[:,:-1])
X_train,X_test,y_train,y_test=train_test_split(scalar_df,df.labal_,test_size=0.3,random_state=42)
linear_svc=SVC(random_state=42,kernel='linear')
linear_svc.fit(X_train,y_train)
print(linear_svc.score(X_test,y_test))

linear_svc_poly=SVC(random_state=42,kernel='poly')
linear_svc_poly.fit(X_train,y_train)
print(linear_svc_poly.score(X_test,y_test))

linear_svc_rbf=SVC(random_state=42,kernel='rbf')
linear_svc_rbf.fit(X_train,y_train)
print(linear_svc_rbf.score(X_test,y_test))
# Scale all the independent features and split the dataset into training and testing set.
param_dict={'C':[0.001,0.01,0.1,1,100,1000],'kernel':['linear','poly','rbf'],'gamma':[0.001,0.01,0.1,1,100,1000]}
svc_gs=GridSearchCV(SVC(),param_grid=param_dict,scoring='accuracy')
svc_gs.fit(X_train,y_train)
svc_gs.score(X_test,y_test) 
# Build model with SVC classifier keeping default Linear kernel and calculate accuracy score.


# Build SVC classifier model with polynomial kernel and calculate accuracy score


# Build SVM model with rbf kernel.


#  Remove Correlated Features.


# Split the newly created data frame into train and test set, scale the features and apply SVM model with rbf kernel to newly created dataframe


# Do Hyperparameter Tuning using GridSearchCV and evaluate the model on test data.





