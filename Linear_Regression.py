import os
import pandas as pd
import numpy as np
import seaborn as sns

# Increase the print output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set working directory
os.chdir("C:/Users/Downloads/IMR/Data_Linear_Regression/")

# Read in the data
rawDf = pd.read_csv("PropertyPrice_Data.csv")
predictionDf = pd.read_csv("PropertyPrice_Prediction.csv")


# Before we do anything, we need to divide our Raw Data into train and test sets (validation)
from sklearn.model_selection import train_test_split
trainDf, testDf = train_test_split(rawDf, train_size=0.8, random_state = 150)

# Create Source Column in both Train and Test
trainDf['Source'] = "Train"
testDf['Source'] = "Test"
predictionDf['Source'] = "Prediction"

# Combine Train and Test
fullRaw = pd.concat([trainDf, testDf, predictionDf], axis = 0)
fullRaw.shape


# Lets drop "Id" column from the data as it is not going to assist us in our model
fullRaw = fullRaw.drop(['Id'], axis = 1) 
# fullRaw.drop(['Id'], axis = 1, inplace = True) 

# Check for NAs
fullRaw.isnull().sum()

# Check data types of the variables
fullRaw.dtypes



############################
# Missing value imputation
############################

# Garage variable (Categorical)
tempMode = trainDf["Garage"].mode()[0]
tempMode
# fullRaw.loc[fullRaw["Source"] == "Train", "Garage"].mode()[0]
# fullRaw["Garage"] = fullRaw["Garage"].fillna(tempMode)  
fullRaw["Garage"].fillna(tempMode, inplace = True)      

# Garage_Built_Year (Continuous)
tempMedian = trainDf["Garage_Built_Year"].median()
tempMedian
fullRaw["Garage_Built_Year"] = fullRaw["Garage_Built_Year"].fillna(tempMedian)   

# All NAs should be gone now
fullRaw.isnull().sum()

############################
# Correlation check
############################

corrDf = fullRaw[fullRaw["Source"] == "Train"].corr()
# corrDf.head()
sns.heatmap(corrDf, 
        xticklabels=corrDf.columns,
        yticklabels=corrDf.columns, cmap='coolwarm_r')


############################
# Dummy variable creation
############################

fullRaw2 = pd.get_dummies(fullRaw, drop_first = True) 
# 'Source'  column will change to 'Source_Train' & 'Source_Test' and it contains 0s and 1s

fullRaw2.shape
fullRaw.shape


############################
# Sampling
############################

# Step 1: Divide into Train, Test and Prediction Dfs
trainDf = fullRaw2[fullRaw2['Source_Train'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
testDf = fullRaw2[fullRaw2['Source_Test'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()
predictionDf = fullRaw2[(fullRaw2['Source_Train'] == 0) & 
                        (fullRaw2['Source_Test'] == 0)].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

trainDf.shape
testDf.shape
predictionDf.shape

# Step 2: Divide into Xs (Indepenedents) and Y (Dependent)
trainX = trainDf.drop(['Sale_Price'], axis = 1).copy()
trainY = trainDf['Sale_Price'].copy()
testX = testDf.drop(['Sale_Price'], axis = 1).copy()
testY = testDf['Sale_Price'].copy()

trainX.shape
trainY.shape
testX.shape
testY.shape

############################
# Add Intercept Column
############################

# In Python, linear regression function does NOT account for an intercept by default.
# So, we need to explicitely add intercept (in the df) - a column called "const" with all values being 1 in it.
from statsmodels.api import add_constant
trainX = add_constant(trainX)
testX = add_constant(testX)
predictionDf = add_constant(predictionDf)

trainX.shape
testX.shape

#########################
# VIF check
#########################

from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # The VIF that will be calculated at EVERY iteration in while loop
maxVIF = 5
trainXCopy = trainX.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIF):
    
    print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    # tempColumnName = tempVIFDf.sort_values(["VIF"])[-1:]["Column_Name"].values[0]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    # tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIF): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is intercept.
highVIFColumnNames

trainX = trainX.drop(highVIFColumnNames, axis = 1)
testX = testX.drop(highVIFColumnNames, axis = 1)
predictionDf = predictionDf.drop(highVIFColumnNames, axis = 1)

trainX.shape
testX.shape


#########################
# Model Building
#########################

from statsmodels.api import OLS
m1ModelDef = OLS(trainY, trainX) # (Dep_Var, Indep_Vars) # This is model definition
m1ModelBuild = m1ModelDef.fit() # This is model building
m1ModelBuild.summary() # This is model output summary

# Extract/ Identify p-values from model
dir(m1ModelBuild)
m1ModelBuild.pvalues

#########################
# Model Optimization
#########################

# Unlike linear regression in R, we dont have a "step()" function.
# We will use a for loop and discard indep variables based on "p-value"
# The concept of the for loop will remain very similar to VIF loop.

tempMaxPValue = 0.1
maxPValue = 0.1
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValue):
    
    print(counter)    
    
    tempModelDf = pd.DataFrame()    
    Model = OLS(trainY, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) # If there is some calculation error resulting in NAs
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValue): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames

# Check final model summary
Model.summary()
trainX = trainX.drop(highPValueColumnNames, axis = 1)
testX = testX.drop(highPValueColumnNames, axis = 1)
predictionDf = predictionDf.drop(highPValueColumnNames, axis = 1)

trainX.shape
testX.shape

# Build model on trainX, trainY (after removing insignificant columns)
Model = OLS(trainY, trainX).fit()
Model.summary()

#########################
# Model Prediction
#########################


Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testY[:6]

#########################
# Model Diagnostics
#########################

import seaborn as sns
# Few checks

# Homoskedasticity check
sns.scatterplot(Model.fittedvalues, Model.resid) # Should not show prominent non-constant variance (heteroskadastic) of errors against fitted values

# Normality of errors check
sns.distplot(Model.resid) # Should be somewhat close to normal distribution

#########################
# Model Evaluation
#########################

# RMSE
np.sqrt(np.mean((testY - Test_Pred)**2))
# This means on an "average", the house price prediction would have +/- error of about 56140


# MAPE
(np.mean(np.abs(((testY - Test_Pred)/testY))))*100

predictionDf["Predicted_Sale_Price"] = Model.predict(predictionDf.drop(["Sale_Price"], axis = 1))



