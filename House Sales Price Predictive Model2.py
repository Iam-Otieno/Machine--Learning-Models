#!/usr/bin/env python
# coding: utf-8

# # House Sale Price Prediction Model
# This model development is aimed at helping home buyers predict the price range of houses based on their taste or
# characteristic preferences. 
# The taste or characteristic preferences form our independent variables, while the sales prices become the target or dependent variable.
# The project is a school assignment for the Open University of Kenya Data Science class under Dr. Milgo.
# The Random Forest Regression algorithm is employed in the development of this model

# # 1. Importing all necessary libraries and modules

# In[419]:


pip install XGboost


# In[427]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # 2. Loading the dataset

# In[5]:


# Initializing the dataloading function

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df
    


# In[6]:


filepath = "Datasets/House_sales_Prices.csv"
hprice = load_dataset(filepath)


# In[7]:


hprice1 = hprice.copy 


# In[8]:


hprice


# # 3. Inspecting and understanding the dataset

# In[10]:


# Initializing Inpection function

def inspect_dataset(df):
    """
    inspecting and understanding various aspects od the dataset:
    dataset information, dataset dataset shape, datset stats summaries
    null values and duplicates
    """
    print("\n General dataset information")
    print(df.info())

    print("\n Dataset Statistical Summaries")
    print(df.describe())

    print("\n Missing Values per Column")
    print(df.isna().sum().tolist())

    print("\n Dataset Columns")
    print(df.columns.tolist())


# In[11]:


inspect_dataset(hprice)


# # 4. Data Preprocessing

# A. Data Cleaning

# In[14]:


# Imputing Null Values
# Initializing the null value imputation function

def null_impute(df):
    """
    calculating % of missing values first
    The threshold for imputation is set at 60%
    columns with more than 60% null values are dropped.
    Numerical null values imputed by media, more robust than mean
    Categorical null values are imputed by the mode
    """
    missing_percent = df.isna().sum()/len(hprice)*100
    threshold = 60
    columns_to_drop = missing_percent[missing_percent>threshold].index
    df1 = df.drop(columns = columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} columns due to excessive missing values")

    #imputing null values in remaining columns with median
    df2 = df1.fillna(df1.mean(numeric_only = True))

    # imputing caregorical null values in remaining columns with the mode
    df3 = df2.fillna(df2.mode().iloc[0])
    return df3


# In[15]:


hprice1 = null_impute(hprice) 

"""
Dropping columns with excessive null values
Imputing numeric null values with the mean
Imputing categorical null values with the mode
"""


# In[16]:


hprice1


# In[17]:


hprice1.isna().sum().tolist() # Checking for presence of null values


# In[18]:


hprice1.shape


# In[19]:


hprice1["MSZoning"].unique().tolist() 

# Checking a column for uniqueness


# B. Checking for duplicates on the dataset

# In[248]:


hprice1.duplicated().sum()


# In[ ]:


# The dataset has zero duplicates apparently


# C. Checking for and Handling outliers

# In[261]:


numeric_cols = hprice1.select_dtypes(include = ["number"]) # Selecting the numerical columns


# In[271]:


numeric_cols = numeric_cols.drop(columns = 'Id') # Dropping the Id Column


# In[273]:


numeric_cols


# In[277]:


# # Visualizing Outliers using boxplots

# plt.figure(figsize = (5,3))
# sns.boxplot(x = numeric_cols)
# plt.show


# In[285]:


plt.figure(figsize=(10, 6))
sns.boxplot(numeric_cols)
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.title("Boxplots of Multiple Numerical Features")
plt.show()


# In[291]:


# Plotting boxplot using Panads

numeric_cols.boxplot(figsize = (10,6))
plt.title("Boxplots of Selected Features")
plt.xticks(rotation=90)
plt.show()


# In[319]:


# Log Transformation
hprice1_log = hprice1.copy()

numeric_cols = hprice1_log.select_dtypes(include = ["number"]).columns
hprice1_log[numeric_cols] = hprice1_log[numeric_cols].apply(lambda x: np.log1p(x))
hprice1_log.head()


# In[346]:


numeric_cols = hprice1_log.select_dtypes(include = ["number"]).columns


# In[323]:


numeric_cols.boxplot(figsize = (10,6))
plt.title("Boxplots of Selected Features")
plt.xticks(rotation=90)
plt.show()


# In[333]:


# Using IQR to remove the outliers
Q1 = hprice1_log[numeric_cols].quantile(0.25)
Q3 = hprice1_log[numeric_cols].quantile(0.75)

# Compute IQR
IQR = Q3 - Q1

# Define acceptable range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data within range
hprice1_cleaned = hprice1_log[
~((hprice1_log[numeric_cols] < lower_bound) | (hprice1_log[numeric_cols] > upper_bound)).any(axis=1)]

print(f"Original dataset size: {hprice1_log.shape[0]}")
print(f"Cleaned dataset size: {hprice1_cleaned.shape[0]}")


# In[340]:


# Going back to original. The data loss in IQR is to excessive

hprice1_log.drop(columns = "Id", inplace = True)


# In[342]:


hprice1_log


# In[344]:


# Winsorization (Cap Extreme Values Instead of Removing)

from scipy.stats.mstats import winsorize


# In[348]:


hprice1_winsorized = hprice1_log.copy()
for col in numeric_cols:
    hprice1_winsorized[col] = winsorize(hprice1_log[col], 
                                        limits=[0.05, 0.05])  # Caps top and bottom 5%


# In[350]:


# Visualize after Winsorization
plt.figure(figsize=(12, 6))
sns.boxplot(data=hprice1_winsorized)
plt.xticks(rotation=90)
plt.title("Boxplot after Winsorization")
plt.show()


# In[352]:


hprice1_winsorized.shape


# In[364]:


# Working on columns with excessive zero values for normalization
 # Checking dataset distribution
# Selct numerical columns from the dataset
num_cols = hprice1_winsorized.select_dtypes(include = "number").columns

# Set up the figure size and grid layout
num_features = len(num_cols)
rows = (num_features // 3) + (num_features % 3 > 0)  # Create enough rows for 3 columns per row

plt.figure(figsize=(15, rows * 4))  # Adjust figure size dynamically

# Loop through each numerical column and plot
for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, 3, i)
    sns.histplot(hprice1_winsorized[col], bins=50, kde=True, color="blue")
    plt.title(col)

plt.tight_layout()  # Adjust spacing to fit everything nicely
plt.show()


# In[366]:


# Count percentage of zeros per column
zero_percent = (hprice1_winsorized == 0).sum() / len(hprice1_winsorized) * 100

# Display columns with high zero percentage
zero_percent[zero_percent > 10]  # Adjust threshold as needed


# In[368]:


# Drop columns where more than 80% of values are zero
threshold = 80
cols_to_drop = zero_percent[zero_percent > threshold].index
hprice1_winsorized.drop(columns=cols_to_drop, inplace=True)


# In[370]:


hprice1_winsorized.shape


# In[373]:


hprice1_winsorized.columns.tolist()


# # 5. Splitting Dataset into Training and Testing

# In[381]:





# In[383]:


cols_category


# In[399]:


# Splitting into X and y

X = hprice1_winsorized.drop(columns = ["SalePrice"])
y = hprice1_winsorized["SalePrice"]


# In[403]:


# Splitting into training and testing

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[405]:


print("X shape:", X.shape)
print("y shape:", y.shape)


# In[389]:


type(X)


# In[407]:


type(y)


# In[413]:


# Encoding the categorical variables

# Selecting categorical variables
cols_category = hprice1_winsorized.select_dtypes(include = ["object", "category"])

# Initializing Encoder
encoder = ce.TargetEncoder(cols = cols_category )


# In[415]:


# Encoding X_train to avoid data leakage

X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)


# In[417]:


X_train_encoded #Dataset Encoded


# In[423]:


# Scaling 

# Initializing scaler
scaler = StandardScaler()

# Fit on training data & transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Convert back to DataFrame for better readability
import pandas as pd
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Check the scaled data
X_train_scaled.head()


# # 6. Training of models
# Training is done using RandoForestRegressor
# GradientBoostingRegressor
# XGboostRegressor

# In[438]:


# Initializing models
rf_regressor = RandomForestRegressor(
    n_estimators = 1000,
    max_depth = 100,
    n_jobs = -1,
    random_state = 42
)
rf_regressor


# In[452]:


# GradientBoostingRegressor

gbr_regressor = GradientBoostingRegressor(
    n_estimators=1000, 
    learning_rate=0.1, 
    random_state=42
)
gbr_regressor


# In[454]:


# XGboost Regressor 
xgb_regresor = XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.1, 
    random_state=42,
    n_jobs = -1
)
xgb_regresor


# In[ ]:


# Training models


# In[460]:


rf_regressor.fit(X_train_scaled, y_train)


# In[456]:


xgb_regresor.fit(X_train_scaled, y_train)


# In[458]:


gbr_regressor.fit(X_train_scaled, y_train)


# In[464]:


# Predictions
#RandomForestRegressor
rf_preds = rf_regressor.predict(X_test_scaled)


# In[466]:


#GradientBoosting

gbr_preds = gbr_regressor.predict(X_test_scaled)


# In[470]:


#XGboost

xgb_preds = xgb_regresor.predict(X_test_scaled)


# In[472]:


# Models Evaluation

# Define evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"🔹 {model_name} Performance:")
    print(f"   - MAE: {mae:.4f}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - R² Score: {r2:.4f}")
    print("-" * 40)

# Evaluate each model
evaluate_model(y_test, rf_preds, "Random Forest")
evaluate_model(y_test, gbr_preds, "Gradient Boosting")
evaluate_model(y_test, xgb_preds, "XGBoost")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




