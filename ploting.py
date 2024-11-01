# %% [markdown]
# #  House Price Pridection

# %% [markdown]
# ## 1. Data Preprocessing

# %% [markdown]
# ### 1.1 Importing Liberaries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# %% [markdown]
# ### 1.2 Data Acquisition and DataFrame Creation

# %%
df = pd.read_csv('../CSV/India.csv', index_col=False)

# %% [markdown]
# ### 1.3 Data Evaluation

# %%
df.head()

# %%
df.size

# %%
df.shape

# %%
df.info()

# %%
df.columns

# %%
df.isnull().sum()

# %%
df.describe()

# %% [markdown]
# ### 1.4 Splitting Data: Dependent and Independent Variables

# %%
# Here df.sample is used to sample out data and assign it to df1 (frac = 1 states that 100% of data from df is sampled and assigned to df1).
# If df1 = df would be used, then changes made in df1 will be replicated into df.
# df is used for preserving global status of csv file.

df1 = df.sample(frac = 1)
df1.shape

# %%
data = df1.drop('Price', axis = 1)
data.head()

# %%
price = df1['Price']
price.head()

# %% [markdown]
# ## 2. Data Transfromation and Scaling

# %% [markdown]
# ### 2.1 Encoding Data: Label Encoding

# %%
# Checking that, is it feasible to use Label Encoding for 'Location column'
# ---> Yes, Because the no.of Total Rows are 7719 and No. of unique values for Location are 413.

df['Location'].unique()
df['Location'].unique().size

# %%
df['City'].unique()
df['City'].unique().size

# %%
# from sklearn.preprocessing ---> LabelEncoder

label_encoder = LabelEncoder()
print(label_encoder)

# %%
df_le = df1.copy()
df_le['Location'] = label_encoder.fit_transform(df['Location'])

# %%
df_le.head()

# %%
df_le['Location'].info()

# %%
# sns.scatterplot(df_le['Price'])
# plt.show()

# %%
plt.figure(figsize= (40,25))
sns.heatmap(df_le.corr(), annot=True)
# plt.show()
plt.savefig('../Results/Corelation-heatmap.png', dpi=500, bbox_inches='tight')

# %% [markdown]
# ### 2.2 Encoding Data: One Hot Encoding

# %%


# %%
# from sklearn.preprocessing ---> OneHotEncoder

# df_ohe = df1.copy()
# ohe = OneHotEncoder()
# print(ohe)

# %%
# feature_array = ohe.fit_transform(df_ohe[['Location']]).toarray()

# %%
# feature_array = feature_array[:,1:]
# column_names = df_ohe['Location'].tolist()

# %%
# df_ohe.drop('Location', axis = 1)
# pd.DataFrame(feature_array, columns = column_names)

# %% [markdown]
# ### 2.3 Data Scaling

# %%
# scaler = MinMaxScaler()




# %%
