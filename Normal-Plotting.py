#!/usr/bin/env python
# coding: utf-8

# # BE Final Project: House Price Pridection

# ## 1. Data Preprocessing

# ### 1.1 Importing Required Liberaries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix
from sklearn import metrics


# ### 1.2 Data Acquisition and DataFrame Creation

# In[2]:


df = pd.read_csv('../CSV/India.csv', index_col=False)


# ### 1.3 Data Evaluation

# In[3]:


df.head()


# In[4]:


df.size


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.columns


# ### 1.4 Data Insights

# In[8]:


df.isnull().sum()


# In[9]:


df.describe()['Wardrobe']


# In[10]:


# Corelation without considerating LOCATION.

df.corr()


# In[11]:


plt.figure(figsize= (40,25))
sns.heatmap(df.corr(), annot=True)
plt.title("Corelation Without LOCATION Factor")
plt.savefig('../Results/Corelation-heatmap-Corelation-Without-LOCATION-Factor', dpi=500, bbox_inches='tight')


# In[ ]:





# ## 2. Data Transfromation and Scaling

# ### 2.1 Encoding Data: Label Encoding

# In[12]:


# Checking that, is it feasible to use Label Encoding for 'Location column'?
# ---> Yes, Because the no.of Total Rows are 7719 and No. of unique values for Location are 413.

df['Location'].unique()
df['Location'].unique().size


# In[13]:


df['City'].unique()
df['City'].unique().size


# In[14]:


# from sklearn.preprocessing ---> LabelEncoder

label_encoder = LabelEncoder()
print(label_encoder)


# In[15]:


df_encoded = df.copy()
df_encoded['Location'] = label_encoder.fit_transform(df['Location'])
df_encoded['City'] = label_encoder.fit_transform(df['City'])


# In[16]:


df_encoded.head()


# In[17]:


df_encoded['Location'].info()


# In[18]:


df_encoded['City'].info()


# In[19]:


sns.scatterplot(df_encoded['Price'])
plt.show()


# In[20]:


plt.figure(figsize= (40,25))
sns.heatmap(df_encoded.corr(), annot=True)
plt.title("Corealtion With LOCATION Factor")
# plt.show()
plt.savefig('../Results/Corelation-heatmap-Corelation-With-LOCATION-Factor', dpi=500, bbox_inches='tight')


# In[21]:


df_encoded.corr().iloc[0].sort_values(ascending=False)


# ### 2.2 Encoding Data: One Hot Encoding

# In[ ]:





# In[22]:


# from sklearn.preprocessing ---> OneHotEncoder

# df_ohe = df1.copy()
# ohe = OneHotEncoder()
# print(ohe)


# In[23]:


# feature_array = ohe.fit_transform(df_ohe[['Location']]).toarray()


# In[24]:


# feature_array = feature_array[:,1:]
# column_names = df_ohe['Location'].tolist()


# In[25]:


# df_ohe.drop('Location', axis = 1)
# pd.DataFrame(feature_array, columns = column_names)


# ### 2.3 Data Scaling

#  Scaling AREA and PRICE Data as it contains high deviation of values

# In[26]:


df_scaled = df_encoded.copy()
scaler = MinMaxScaler()
df_scaled[['Area']] = scaler.fit_transform(df_scaled[['Area']])
df_scaled[['Price']] = scaler.fit_transform(df_scaled[['Price']])


# In[27]:


df_scaled.head()


# ### 2.4 Splitting Data: Dependent and Independent Variable

# In[28]:


df_temp = df_scaled.copy()
data = df_temp.drop('Price', axis=1)
price = df_temp['Price']


# In[29]:


data.head()


# In[30]:


price.head()


# ### 2.5 Train - Test Split

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(data, price, test_size=0.2)


# In[32]:


print(len(X_train), len(X_test), len(y_train), len(y_test))


# ## 3 Model Evaluation

# ### 3.1 Linear Regression

# #### 3.1.1 Linear Regression: Training Model

# In[33]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[34]:


LR = LinearRegression()
LR.fit(X_train, y_train)


# #### 3.1.2 Linear Regression: Testing

# In[35]:


y_predict_LR = LR.predict(X_test)
print(y_predict_LR)


# #### 3.1.3 Linear Regression: Score / Results

# In[36]:


LR_score = LR.score(X_test, y_test)
print(LR_score*100)


# In[37]:


R2_Score_LR = r2_score(y_test, y_predict_LR)*100
MAE_LR = metrics.mean_absolute_error(y_test, y_predict_LR)*100
MSE_LR = metrics.mean_squared_error(y_test, y_predict_LR)*100
RSME_LR = np.sqrt(metrics.mean_squared_error(y_test, y_predict_LR))*100
print('R2 Score Linear Regression', R2_Score_LR)
print('MAE Linear Regression:', MAE_LR)
print('MSE Linear Regression:', MSE_LR)
print('RMSE Linear Regression:', RSME_LR)


# In[38]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_LR, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Linear Regression: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Linear-Regression-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[116]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_LR)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Linear Regression: Original Prices vs Predicted Prices")
plt.savefig('../Results/Linear-Regression-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')

# ### 3.2 Ridge Regression

# #### 3.2.1 Redge Regression: Model Training

# In[39]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[40]:


RR = Ridge()
RR.fit(X_train,y_train)


# #### 3.2.2 Ridge Regression: Testing

# In[41]:


y_predict_RR = RR.predict(X_test)
print(y_predict_RR)


# #### 3.2.3 Ridge Regression: Score / Results

# In[42]:


RR_score = RR.score(X_test, y_test)
print(RR_score*100)


# In[43]:


R2_Score_RR = r2_score(y_test, y_predict_RR)*100
MAE_RR = metrics.mean_absolute_error(y_test, y_predict_RR)*100
MSE_RR = metrics.mean_squared_error(y_test, y_predict_RR)*100
RSME_RR = np.sqrt(metrics.mean_squared_error(y_test, y_predict_RR))*100
print('R2 Score Linear Regression', R2_Score_RR)
print('MAE Ridge Regression:', MAE_RR)
print('MSE Ridge Regression:', MSE_RR)
print('RMSE Ridge Regression:', RSME_RR)


# In[44]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_RR, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Ridge Regression: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Ridge-Regression-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[117]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_RR)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Ridge Regression: Original Prices vs Predicted Prices")
plt.savefig('../Results/Ridge-Regression-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.3 Lasso Regression

# #### 3.3.1 Lasso Regression: Model Training

# In[45]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[46]:


Lasso_R = Lasso()
Lasso_R.fit(X_train,y_train)


# #### 3.3.2 Lasso Regression: Testing

# In[47]:


y_predict_Lasso_R = Lasso_R.predict(X_test)
print(y_predict_Lasso_R)


# #### 3.3.3 Lasso Regression: Score / Results

# In[48]:


Lasso_R_score = Lasso_R.score(X_test, y_test)
print(Lasso_R_score*100)


# In[49]:


R2_Score_Lasso_R = r2_score(y_test, y_predict_Lasso_R)*100
MAE_Lasso_R = metrics.mean_absolute_error(y_test, y_predict_Lasso_R)*100
MSE_Lasso_R = metrics.mean_squared_error(y_test, y_predict_Lasso_R)*100
RSME_Lasso_R = np.sqrt(metrics.mean_squared_error(y_test, y_predict_Lasso_R))*100
print('R2 Score Linear Regression', R2_Score_Lasso_R)
print('MAE Ridge Regression:', MAE_Lasso_R)
print('MSE Ridge Regression:', MSE_Lasso_R)
print('RMSE Ridge Regression:', RSME_Lasso_R)


# In[50]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_Lasso_R, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Lasso Regression: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Lasso-Regression-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[118]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_Lasso_R)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Lasso Regression: Original Prices vs Predicted Prices")
plt.savefig('../Results/Lasso-Regression-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.4 Decision Tree

# #### 3.4.1 Decision Tree: Model Training

# In[51]:


from sklearn.tree import DecisionTreeRegressor


# In[52]:


DT = DecisionTreeRegressor(max_depth=5)
DT.fit(X_train,y_train)


# #### 3.4.2 Decision Tree: Testing

# In[53]:


y_predict_DT = DT.predict(X_test)
print(y_predict_DT)


# #### 3.4.3 Decision Tree: Score / Results

# In[54]:


DT_score = DT.score(X_test, y_test)
print(DT_score*100)


# In[55]:


R2_Score_DT = r2_score(y_test, y_predict_DT)*100
MAE_DT = metrics.mean_absolute_error(y_test, y_predict_DT)*100
MSE_DT = metrics.mean_squared_error(y_test, y_predict_DT)*100
RSME_DT = np.sqrt(metrics.mean_squared_error(y_test, y_predict_DT))*100
print('R2 Score Linear Regression', R2_Score_DT)
print('MAE Ridge Regression:', MAE_DT)
print('MSE Ridge Regression:', MSE_DT)
print('RMSE Ridge Regression:', RSME_DT)


# In[56]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_DT, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Decision Tree: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Decision-Tree-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[57]:


from sklearn import tree


# In[58]:


plt.figure(figsize= (40,25))
tree.plot_tree(DT, feature_names=data.columns, class_names=price, filled = True)
plt.title("Decision Tree Depiction")
plt.savefig('../Results/Decision-Tree-Depiction', dpi=500, bbox_inches='tight')


# In[59]:


# !pip install -q dtreeviz -> for visualization not working currently


# In[119]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_DT)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Decision Tree: Original Prices vs Predicted Prices")
plt.savefig('../Results/Decision-Tree-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.5 Random Forest

# #### 3.5.1 Random Forest: Model Training

# In[60]:


from sklearn.ensemble import RandomForestRegressor


# In[61]:


RF = RandomForestRegressor(n_estimators=280, random_state=0)
RF.fit(X_train, y_train)


# #### 3.5.2 Random Forest: Testing

# In[62]:


y_predict_RF = RF.predict(X_test)
print(y_predict_RF)


# #### 3.5.3 Random Forest: Score / Results

# In[63]:


RF_score = RF.score(X_test, y_test)
print(RF_score*100)


# In[64]:


R2_Score_RF = r2_score(y_test, y_predict_RF)*100
MAE_RF = metrics.mean_absolute_error(y_test, y_predict_RF)*100
MSE_RF = metrics.mean_squared_error(y_test, y_predict_RF)*100
RSME_RF = np.sqrt(metrics.mean_squared_error(y_test, y_predict_RF))*100
print('R2 Score Linear Regression', R2_Score_RF)
print('MAE Ridge Regression:', MAE_RF)
print('MSE Ridge Regression:', MSE_RF)
print('RMSE Ridge Regression:', RSME_RF)


# In[65]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_RF, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Random Forest: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Random-Forest-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[66]:


# Confusion Matrix
# cm_RF = confusion_matrix(y_test, y_predict_RF)
# cm_RF


# In[120]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_RF)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Random Forest: Original Prices vs Predicted Prices")
plt.savefig('../Results/Random-Forest-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.6 K Nearest Neighbour: KNN

# #### 3.6.1 KNN: Model Training

# In[67]:


from sklearn.neighbors import KNeighborsRegressor


# In[68]:


KNR = KNeighborsRegressor(100)
KNR.fit(X_train, y_train)


# #### 3.6.2 KNN: Testing

# In[69]:


y_predict_KNR = KNR.predict(X_test)
print(y_predict_KNR)


# #### 3.6.3 KNN: Score / Results

# In[70]:


KNR_score = KNR.score(X_test, y_test)
print(KNR_score*100)


# In[71]:


R2_Score_KNR = r2_score(y_test, y_predict_KNR)*100
MAE_KNR = metrics.mean_absolute_error(y_test, y_predict_KNR)*100
MSE_KNR = metrics.mean_squared_error(y_test, y_predict_KNR)*100
RSME_KNR = np.sqrt(metrics.mean_squared_error(y_test, y_predict_KNR))*100
print('R2 Score Linear Regression', R2_Score_KNR)
print('MAE Ridge Regression:', MAE_KNR)
print('MSE Ridge Regression:', MSE_KNR)
print('RMSE Ridge Regression:', RSME_KNR)


# In[72]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_KNR, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("KNN: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/KNN-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[121]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_KNR)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("KNN: Original Prices vs Predicted Prices")
plt.savefig('../Results/KNN-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.7 Support Vector Machine: SVM

# #### 3.7.1 SVM: Model Training

# In[73]:


from sklearn.svm import SVR


# In[74]:


SVM = SVR()
SVM.fit(X_train,y_train)


# #### 3.7.2 SVM: Testing

# In[75]:


y_predict_SVM = SVM.predict(X_test)
print(y_predict_SVM)


# #### 3.7.3 SVM: Score / Results

# In[76]:


SVM_score = SVM.score(X_test, y_test)
print(SVM_score*100)


# In[77]:


R2_Score_SVM = r2_score(y_test, y_predict_SVM)*100
MAE_SVM = metrics.mean_absolute_error(y_test, y_predict_SVM)*100
MSE_SVM = metrics.mean_squared_error(y_test, y_predict_SVM)*100
RSME_SVM = np.sqrt(metrics.mean_squared_error(y_test, y_predict_SVM))*100
print('R2 Score Linear Regression', R2_Score_SVM)
print('MAE Ridge Regression:', MAE_SVM)
print('MSE Ridge Regression:', MSE_SVM)
print('RMSE Ridge Regression:', RSME_SVM)


# In[78]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_SVM, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("Support Vector Machine: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/Support-Vector-Machine-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')


# In[122]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_SVM)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("Support Vector Machine: Original Prices vs Predicted Prices")
plt.savefig('../Results/Support-Vector-Machine-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ### 3.8 Extreme Gradient Boost: XGBoost

# #### 3.8.1 XGBoost: Model Training

# In[79]:


from xgboost import XGBRegressor


# In[80]:


XGB = XGBRegressor()
XGB.fit(X_train, y_train)


# #### 3.8.2 XGBoost: Testing

# In[81]:


y_predict_XGB = XGB.predict(X_test)
print(y_predict_XGB)


# #### 3.8.3 XGBoost: Score / Results

# In[82]:


XGB_score = XGB.score(X_test, y_test)
print(XGB_score*100)


# In[83]:


R2_Score_XGB = r2_score(y_test, y_predict_XGB)*100
MAE_XGB = metrics.mean_absolute_error(y_test, y_predict_XGB)*100
MSE_XGB = metrics.mean_squared_error(y_test, y_predict_XGB)*100
RSME_XGB = np.sqrt(metrics.mean_squared_error(y_test, y_predict_XGB))*100
print('R2 Score Linear Regression', R2_Score_XGB)
print('MAE Ridge Regression:', MAE_XGB)
print('MSE Ridge Regression:', MSE_XGB)
print('RMSE Ridge Regression:', RSME_XGB)


# In[84]:


plt.figure(figsize= (40,25))
plt.scatter(y_predict_XGB, y_test)
# plt.plot(y_test, LR.predict(X_test), color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Original prices")
plt.title("XGBoost: Original Prices vs Predicted Prices")
# plt.show()
plt.savefig('../Results/XGBoost-Predicted-vs-Original-Scattered', dpi=500, bbox_inches='tight')



# In[123]:


x = np.sort(y_test)[:-7]
y = np.sort(y_predict_XGB)[:-7]

plt.figure(figsize=(25,20))
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')
plt.title("XGBoost: Original Prices vs Predicted Prices")
plt.savefig('../Results/XGBoost-Predicted-vs-Original-Line', dpi=500, bbox_inches='tight')


# ## 4 Results

# ### 4.1 Score Comparision

# In[88]:


All_result = pd.DataFrame.from_dict({"Models" : ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest", "K Nearest Neighbour", "Support Vector Machine", "XGBoost"],
                                    "Score": [LR_score, RR_score, Lasso_R_score, DT_score, RF_score, KNR_score, SVM_score, XGB_score],
                                    "Mean Absolute Error" : [MAE_LR, MAE_RR, MAE_Lasso_R, MAE_DT, MAE_RF, MAE_KNR, MAE_SVM, MAE_XGB],
                                    "Mean Square Error" : [MSE_LR, MSE_RR, MSE_Lasso_R, MSE_DT, MSE_RF, MSE_KNR, MSE_SVM, MSE_XGB],
                                    "Route Mean Square Error" : [RSME_LR, RSME_RR, RSME_Lasso_R, RSME_DT, RSME_RF, RSME_KNR, RSME_SVM, RSME_XGB]
                                    })
All_result.head(8)


# In[93]:


plt.figure(figsize= (40,25))
sns.barplot(x='Models',y='Route Mean Square Error',data=All_result)
plt.title('Route Mean Square Error Comparision')
plt.savefig('../Results/Route-Mean-Square-Error-Comparision', dpi=500, bbox_inches='tight')


# In[132]:


plt.figure(figsize=(60,25))

#Linear Regression
plt.subplot(2,4,1)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_LR)[:-7]
plt.title("Linear Regression")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#Ridge Regression
plt.subplot(2,4,2)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_RR)[:-7]
plt.title("Ridge Regression")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#Lasso Regression
plt.subplot(2,4,3)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_Lasso_R)[:-7]
plt.title("Lasso Regression")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#Decision Tree
plt.subplot(2,4,4)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_DT)[:-7]
plt.title("Decision Tree")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#Random Forest
plt.subplot(2,4,5)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_RF)[:-7]
plt.title("Random Forest")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#K Nearest Neighbour
plt.subplot(2,4,6)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_KNR)[:-7]
plt.title("K Nearest Neighbour")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#Support Vector Machine
plt.subplot(2,4,7)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_SVM)[:-7]
plt.title("Support Vector Machine")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

#XGBoost
plt.subplot(2,4,8)
x = np.sort(y_test)[:-7]
y = np.sort(y_predict_XGB)[:-7]
plt.title("XGBoost")
plt.plot(x, label = "Actual", color="red")
plt.plot(y, label = "Pridected", color='green')

plt.savefig('../Results/All-Results', dpi=500, bbox_inches='tight')

# In[ ]:




