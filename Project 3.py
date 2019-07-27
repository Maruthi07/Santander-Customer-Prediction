#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb


# In[25]:


train = pd.read_csv("C:/Users/ramaruth/Desktop/Data Scientist/Project 3/train.csv")
test = pd.read_csv("C:/Users/ramaruth/Desktop/Data Scientist/Project 3/test.csv")


# In[26]:


train.head()


# In[27]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[28]:


missing_data(train)


# In[29]:


missing_data(test)


# In[14]:


#checking outliers using Chauvenet's criterion
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array
    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's
    prob = erfc(d)                # Area normal dist.    
    return prob < criterion       # Use boolean array outside this function


# In[15]:


numerical_features=train.columns[2:]


# In[17]:


from scipy.special import erfc
train_outliers = dict()
for col in [col for col in numerical_features]:
    train_outliers[col] = train[chauvenet(train[col].values)].shape[0]
train_outliers = pd.Series(train_outliers)

train_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[19]:


print('Total number of outliers in training set: {} ({:.2f}%)'.format(sum(train_outliers.values), (sum(train_outliers.values) / train.shape[0]) * 100))


# In[20]:


#outliers in each variable in test data 
test_outliers = dict()
for col in [col for col in numerical_features]:
    test_outliers[col] = test[chauvenet(test[col].values)].shape[0]
test_outliers = pd.Series(test_outliers)

test_outliers.sort_values().plot(figsize=(14, 40), kind='barh').set_xlabel('Number of outliers');


# In[21]:


print('Total number of outliers in testing set: {} ({:.2f}%)'.format(sum(test_outliers.values), (sum(test_outliers.values) / test.shape[0]) * 100))


# In[22]:


#remove these outliers in train and test data
for col in numerical_features:
    train=train.loc[(~chauvenet(train[col].values))]
for col in numerical_features:
    test=test.loc[(~chauvenet(test[col].values))]


# In[30]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()


# In[31]:


import seaborn as sns
t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
features = train.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[32]:


# distribution of the mean values per row in the train and test set.
plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[33]:


#distribution of the mean values per columns in the train and test set.
    
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[34]:


# distribution of standard deviation of values per row for train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


# In[35]:


# distribution of the standard deviation of values per columns in the train and test datasets.

plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


# In[36]:


plt.figure(figsize=(16,6))
plt.title("Distribution of skew per row in the train and test set")
sns.distplot(train[features].skew(axis=1),color="red", kde=True,bins=120, label='train')
sns.distplot(test[features].skew(axis=1),color="orange", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[37]:


plt.figure(figsize=(16,6))
plt.title("Distribution of skew per column in the train and test set")
sns.distplot(train[features].skew(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test[features].skew(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# In[38]:


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


# In[39]:


correlations.tail(10)


# In[40]:


import seaborn as sns
sns.factorplot('target', data=train, kind='count')


# In[41]:


#count of both class(number of classes)
train['target'].value_counts()


# In[42]:


train.shape


# In[43]:


#WE seperate the dataset whose target class is belong to class 0
data=train.loc[train['target'] == 0]
#choose starting 24000 rows
data2=data.loc[:24000]
data2


# In[44]:


#WE seperate the dataset whose target class is belong to class 1
data1=train.loc[train['target'] == 1]
data1


# In[45]:


#Add both Dataframe data1 and data2 in one dataframe
newdata=pd.concat([data1, data2], ignore_index=True)
newdata


# In[46]:


#Add both Dataframe data1 and data2 in one dataframe
newdata=pd.concat([data1, data2], ignore_index=True)
newdata


# In[47]:


#Shuffle the Dataframe
newdata=newdata.sample(frac=1)
newdata


# In[49]:


sns.factorplot('target', data=newdata, kind='count')


# In[50]:


#Seperate the input features and store in variable x
x=newdata.iloc[:,2:].values
x=pd.DataFrame(x)
x


# In[51]:


#Seprate the target class and store the class in y variable
y=newdata.iloc[:,1].values
y=pd.DataFrame(y)
y


# In[52]:


#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=100,test_size=0.2)


# In[53]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[54]:


#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=80)
PCA_X_train=pca.fit_transform(X_train)
PCA_X_test=pca.fit_transform(X_test)
explain=pca.explained_variance_ratio_.tolist()
explain


# In[55]:


X_train.shape


# In[56]:


PCA_X_train.shape


# In[57]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression(n_jobs=-1)


# In[58]:



model.fit(PCA_X_train,y_train)


# In[59]:


y_pred=model.predict(PCA_X_test)


# In[60]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[61]:


#find precision ,recall,fscore,support
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[62]:


(3441+3001)/(3441+917+986+3001)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1500,random_state=0)
classifier.fit(PCA_X_train,y_train)


# In[64]:



#Predict from test data
y_pred=classifier.predict(PCA_X_test)


# In[65]:


#Appliying confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[66]:


(3497+2970)/(3497+861+1017+2970)


# In[67]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred, average='binary')


# In[68]:


#find precision ,recall,fscore,support
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))


# In[69]:


#Applying Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[70]:


test_data=pd.read_csv("C:/Users/ramaruth/Desktop/Data Scientist/Project 3/test.csv")


# In[71]:


sc.fit(test_data.iloc[:,1:])


# In[61]:


test_data_std=pd.DataFrame(sc.transform(test_data.iloc[:,1:]))


# In[63]:


pca.fit(test_data_std)


# In[65]:


test_data_X=pd.DataFrame(pca.transform(test_data_std))


# In[67]:


(pd.DataFrame(model.predict(test_data_X))).shape


# In[69]:


predictions=model.predict(test_data_X)


# In[71]:


test_data_X['target']=predictions


# In[73]:



test_data_X.head()


# In[74]:


test_data_X.to_csv('test_data_X.csv')

