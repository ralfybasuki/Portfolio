#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train_data=pd.read_excel(r"C:\Users\ralfy\OneDrive\Desktop\CSV Files\Data_Train.xlsx")


# In[3]:


train_data.head()


# In[4]:


train_data.copy()


# In[5]:


df = train_data.copy()


# In[6]:


df


# In[7]:


pd.options.display.max_rows = None


# In[8]:


df


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# ### Dropping Null Values

# In[11]:


df.dropna(inplace = True)


# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# ### Defining date and time function

# In[14]:


def change_into_datetime(col):
    df[col]=pd.to_datetime(df[col])


# In[15]:


df.columns


# ### Change date and time data types to date and time function

# In[16]:


for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
      change_into_datetime(i)


# In[17]:


df.dtypes


# ### Separate the Day and Month

# In[18]:


df['Journey_Day'] = df['Date_of_Journey'].dt.day


# In[19]:


df['Journey_Month'] = df['Date_of_Journey'].dt.month


# In[20]:


df.head()


# ### Drop the Date_of_Journey column as it is not necessary anymore

# In[21]:


df.drop('Date_of_Journey', axis=1, inplace=True)


# In[22]:


df.head()


# ### Defining new functions to extract hours, minutes, and drop columns from the Dep_Time

# In[23]:


def extract_min(df,col):
     df[col+'_min']=df[col].dt.hour


# In[24]:


def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)


# In[25]:


def extract_hour(df,col):
     df[col+'_hour']=df[col].dt.hour


# In[26]:


extract_hour(df,'Dep_Time')


# In[27]:


extract_min(df,'Dep_Time')


# In[28]:


drop_column(df,'Dep_Time')


# In[29]:


df.head()


# ### Extract hours, minutes, and drop column from the Arrival_Time

# In[30]:


extract_hour(df,'Arrival_Time')


# In[31]:


extract_min(df,'Arrival_Time')


# In[32]:


drop_column(df,'Arrival_Time')


# In[33]:


df.head()


# ### Split the duration

# In[34]:


'2h 50m'.split(' ')


# In[35]:


duration=list(df['Duration'])


# ### Adding 0h and 0m to the duration

# In[36]:


for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:                   # Check if duration contains only hour\n",
                duration[i]=duration[i] + ' 0m'      # Adds 0 minute\n",
        else:
            duration[i]='0h '+ duration[i]       # if duration contains only second, Adds 0 hour\n",
        


# In[37]:


df['Duration'] = duration


# In[38]:


df.head()


# In[39]:


'2h 50m'.split(' ')[1][0:-1]


# ### Defining hour and min function to split the hour and min

# In[40]:


def hour(x):
   return x.split(' ')[0][0:-1]


# In[41]:


def min(x):
   return x.split(' ')[1][0:-1]


# In[42]:


df['Duration_hours']=df['Duration'].apply(hour)
df['Duration_mins']=df['Duration'].apply(min)


# In[43]:


df.head()


# ### Drop the Duration column as it is not necessary anymore

# In[44]:


df.drop('Duration', axis=1, inplace = True )


# In[45]:


df.head()


# ### Check the data types

# In[46]:


df.dtypes


# ### Change Duration_hours and Duration_mins's data type into intergers

# In[47]:


df['Duration_hours'] = df['Duration_hours'].astype(int)
df['Duration_mins'] = df['Duration_mins'].astype(int)


# In[48]:


df.dtypes


# In[49]:


df.head()


# ### Separate between Categorical and Continual data

# In[50]:


cat_col = [col for col in df.columns if df[col].dtype=='O']
cat_col


# In[51]:


cont_col = [col for col in df.columns if df[col].dtype!='O']
cont_col 


# In[52]:


categorical = df[cat_col]
categorical.head()


# ### Count the sum of all airplane in the data

# In[53]:


categorical['Airline'].value_counts()


# ### Create the boxplot about the airplaine price

# In[54]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price', x='Airline',data=df.sort_values('Price', ascending=False))


# ### Create a boxplot about price according to the total stops

# In[55]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price', x='Total_Stops',data=df.sort_values('Price', ascending=False))


# In[56]:


len(categorical['Airline'].unique())


# ### Create a dummy variable in the Airline categorical data

# In[57]:


Airline = pd.get_dummies(categorical['Airline'], drop_first=True)
Airline.head()


# ### Count the sum of Source

# In[58]:


categorical['Source'].value_counts()


# ### Create a catplot regarding the price according to the source

# In[59]:


plt.figure(figsize=(15,5))
sns.catplot(y='Price', x='Source',data=df.sort_values('Price', ascending=False),kind='boxen')


# ### Create a dummy variable in the Source categorical data

# In[60]:


Source = pd.get_dummies(categorical['Source'], drop_first=True)
Source.head()


# In[61]:


categorical['Destination'].value_counts()


# ### Create a dummy variable in the Destination categorical data

# In[62]:


Destination = pd.get_dummies(categorical['Destination'], drop_first=True)
Destination.head()


# ### Split the Route categorical data

# In[63]:


categorical['Route']


# In[64]:


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]


# In[65]:


categorical.head()


# ### Fill the NA Variable in the route columns

# In[66]:


categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)


# In[67]:


categorical.head()


# ### Print the sum of all categorical data

# In[68]:


for feature in categorical.columns:
    print('{} has total {} categories'.format(feature,len(categorical[feature].value_counts())))


# ### Import the Label Encoder from the Sklearn preprocessing library

# In[69]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[70]:


categorical.columns


# ### Input the label encoder to the categorical data

# In[71]:


for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i] = encoder.fit_transform(categorical[i])


# In[72]:


categorical.head()


# ### drop the route and additional_info column as it is not necessary anymore
# 

# In[73]:


drop_column(categorical,'Route')
drop_column(categorical,'Additional_Info')


# In[74]:


categorical.head()


# ### Count the Total_Stops value

# In[75]:


categorical['Total_Stops'].value_counts()


# ### get the Total_Stops unique value

# In[76]:


categorical['Total_Stops'].unique()


# ### Create a dictionary about the Total_Stops

# In[77]:


dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[78]:


categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[79]:


categorical.head()


# In[80]:


df[cont_col]


# ### Concatinate the Categorical and Continuous data

# In[81]:


data_train=pd.concat([categorical,Airline,Source,Destination,df[cont_col]],axis=1)
data_train.head()


# In[82]:


drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')


# In[83]:


data_train.head()


# In[84]:


pd.set_option('display.max_columns', 35)


# In[85]:


data_train.head()


# In[86]:


data_train.columns


# ### Dealing with Outliers

# In[87]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)


# In[88]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# In[89]:


data_train['Price']=np.where(data_train['Price']>=40000, data_train['Price'].median(), data_train['Price'])


# In[90]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# ### Separating Independent and Dependent Data

# In[91]:


X=data_train.drop('Price', axis=1)
X.head()


# In[92]:


y=data_train['Price']
y


# In[93]:


type(X)


# In[94]:


type(y)


# In[95]:


X.isnull().sum()


# In[96]:


y.isnull().sum()


# ### Feature Selection

# #### Why to apply Feature Selection?\n",
#      To select important features to get rid of curse of dimensionality ie..to get rid of duplicate features

# In[97]:


np.array(X)


# In[98]:


np.array(y)


# ### I wanted to find mutual information scores or matrix to get to know about the relationship between all features.

# #### Feature selection using Information Gain

# In[99]:


from sklearn.feature_selection import mutual_info_classif


# In[100]:


mutual_info_classif(np.array(X), np.array(y))


# In[101]:


X.dtypes


# In[102]:


mutual_info_classif(X,y)


# In[103]:


imp=pd.DataFrame(mutual_info_classif(X,y), index=X.columns)
imp


# ### List the most important variables from the most to the least

# In[104]:


imp.columns=['importance']
imp.sort_values(by='importance', ascending=False)


# ### Split the train and test data with sklearn 

# In[121]:


from sklearn.model_selection import train_test_split


# In[122]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[287]:


from sklearn import metrics


# 
# ### Dumping model using Pickle

# In[288]:


import pickle


# In[313]:


def predict(ml_model,dump):
    model=ml_model.fit(X_train,y_train)
    print('Training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('predictions are: \\n {}'.format(y_prediction))
    print('\\n')
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score: {}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction) 
    
    if dump==1:
        file=open(r'C:\Users\ralfy\OneDrive\ドキュメント\Datasets\model.pkl')
        pickle.dump(model,file)


# ### Import Random Forest Class

# In[314]:


from sklearn.ensemble import RandomForestRegressor


# In[315]:


predict(RandomForestRegressor(),1)


# ### Play with Multiple Algorithm

# In[138]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[303]:


predict(DecisionTreeRegressor(),0)


# In[306]:


predict(LinearRegression(),0)


# In[307]:


predict(KNeighborsRegressor(),0)


# In[309]:


predict(RandomForestRegressor(),0)


# In[ ]:





# In[339]:


from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor()
# ### Hyperparameter Turning

# In[330]:


from sklearn.model_selection import RandomizedSearchCV


# #### Assigning Hyperparameters

# In[335]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]
max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]


# In[337]:


random_grid = {
    'n_estimators': n_estimators,
    'max_features': ['auto','sqrt'],
    'max_depth': max_depth,
    'min_samples_split':[5,10,15,100]
}


# In[338]:


random_grid


# In[340]:


rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[341]:


rf_random.fit(X_train,y_train)


# In[342]:


rf_random.best_params_


# In[343]:


prediction=rf_random.predict(X_test)


# In[344]:


sns.distplot(y_test-prediction)


# In[345]:


metrics.r2_score(y_test,prediction)


# In[ ]:




