# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:32:40 2019

@author: mbigdelou
"""
import os
os.chdir(r'C:\Users\mbigdelou\Desktop\Data Mining Project')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.interactive(False)
import seaborn as sns
import statsmodels.api as sm

#==============Import Dataset
df=pd.read_csv('dffinal_second_draft_Return_cat_Added.csv')
df.head()

df.shape
df.describe()
df['Return_Annual'].describe()

df.info()

a = df.isnull().any()
df.notnull().sum()
df = df.dropna(how='all')


#==Ploting the target variable
df2 = df.copy()
df2.sort_values('Return_Annual', ascending = True, inplace= True)
plt.scatter(df2['Return_cat'],df2['Return_Annual'], c='r',marker='*')
plt.show()

##Return Distribution
sns.distplot(df.Return_Annual);
plt.show()

#--
Minus_Gain = sum(df.Return_cat == 'Minus_Gain')
Zero_Gain = sum(df.Return_cat == 'Zero_Gain')
Plus_Gain = sum(df.Return_cat == 'Plus_Gain')
Label=['Minus_Gain','Zero_Gain','Plus_Gain']
Return_Type=pd.Series([Minus_Gain,Zero_Gain,Plus_Gain]).astype(int)
pos = np.arange(len(Return_Type))
len(pos)
plt.xticks(pos, Label)
plt.bar(pos,Return_Type,color='red',edgecolor='black')
plt.show()


#=======PREPROCESSIONG
#==Transforming to log
#list of columns
list(df)

#Log Transformation for variables with large scale 
dfDescribe = df.describe()

df['lnAccPay'] = pd.DataFrame(np.log(df['Accounts Payable']))
df['lnAccReci'] = pd.DataFrame(np.log(df['Accounts Receivable']))
df['lnAddiIncomExpense'] = pd.DataFrame(np.log(df["Add'l income/expense items"]))
df['lnAfterTaxROE'] = pd.DataFrame(np.log(df['After Tax ROE']))
df['lnCapExpend'] = pd.DataFrame(np.log(df['Capital Expenditures']))
df['lnCapSurplus'] = pd.DataFrame(np.log(df['Capital Surplus']))
df['lnCashRatio'] = pd.DataFrame(np.log(df['Cash Ratio']))
df['lnCashCashEqui'] = pd.DataFrame(np.log(df['Cash and Cash Equivalents']))
df['lnChangInven'] = pd.DataFrame(np.log(df['Changes in Inventories']))
df['lnCmonStock'] = pd.DataFrame(np.log(df['Common Stocks']))
df['lnCostRev'] = pd.DataFrame(np.log(df['Cost of Revenue']))
df['lnCurRatio'] = pd.DataFrame(np.log(df['Current Ratio']))
df['lnDeferAssetCharge'] = pd.DataFrame(np.log(df['Deferred Asset Charges']))
df['lnDeferLiabilCharge'] = pd.DataFrame(np.log(df['Deferred Liability Charges']))
df['lnDepreciation'] = pd.DataFrame(np.log(df['Depreciation']))
df['lnEBIT'] = pd.DataFrame(np.log(df['Earnings Before Interest and Tax']))
df['lnEBT'] = pd.DataFrame(np.log(df['Earnings Before Tax']))
df['lnEffectExchRt'] = pd.DataFrame(np.log(df['Effect of Exchange Rate']))
df['lnEquityEarn/LossUnconsolid'] = pd.DataFrame(np.log(df['Equity Earnings/Loss Unconsolidated Subsidiary']))
df['lnFixedAssets'] = pd.DataFrame(np.log(df['Fixed Assets']))
df['lnGoodwill'] = pd.DataFrame(np.log(df['Goodwill']))
df['lnGrossMargin'] = pd.DataFrame(np.log(df['Gross Margin']))
df['lnGrossProfit'] = pd.DataFrame(np.log(df['Gross Profit']))
df['lnIncomeTax'] = pd.DataFrame(np.log(df['Income Tax']))
df['lnIntangAsset'] = pd.DataFrame(np.log(df['Intangible Assets']))
df['lnInterestExpense'] = pd.DataFrame(np.log(df['Interest Expense']))
df['lnInventory'] = pd.DataFrame(np.log(df['Inventory']))
df['lnInvestments'] = pd.DataFrame(np.log(df['Investments']))
df['lnLiabilities'] = pd.DataFrame(np.log(df['Liabilities']))
df['lnLTermDebt'] = pd.DataFrame(np.log(df['Long-Term Debt']))
df['lnLTermInvest'] = pd.DataFrame(np.log(df['Long-Term Investments']))
df['lnMinorityInterest'] = pd.DataFrame(np.log(df['Minority Interest']))
df['lnMiscStocks'] = pd.DataFrame(np.log(df['Misc. Stocks']))
df['lnNetBorrowings'] = pd.DataFrame(np.log(df['Net Borrowings']))
df['lnNetCashFlow'] = pd.DataFrame(np.log(df['Net Cash Flow']))
df['lnNetCFOperating'] = pd.DataFrame(np.log(df['Net Cash Flow-Operating']))
df['lnNetCFFinancing'] = pd.DataFrame(np.log(df['Net Cash Flows-Financing']))
df['lnNetCFInvesting'] = pd.DataFrame(np.log(df['Net Cash Flows-Investing']))
df['lnNetIncome'] = pd.DataFrame(np.log(df['Net Income']))
df['lnNetIncomeAdjus'] = pd.DataFrame(np.log(df['Net Income Adjustments']))
df['lnNetIncomeApplicableCmonShrholdrs'] = pd.DataFrame(np.log(df['Net Income Applicable to Common Shareholders']))
df['lnNetIncomeContOperations'] = pd.DataFrame(np.log(df['Net Income-Cont. Operations']))
df['lnNetReceivables'] = pd.DataFrame(np.log(df['Net Receivables']))
df['lnNonRecurringItem'] = pd.DataFrame(np.log(df['Non-Recurring Items']))
df['lnOperIncome'] = pd.DataFrame(np.log(df['Operating Income']))
df['lnOperMargin'] = pd.DataFrame(np.log(df['Operating Margin']))
df['lnOtherAssets'] = pd.DataFrame(np.log(df['Other Assets']))
df['lnOtherCurrAssets'] = pd.DataFrame(np.log(df['Other Current Assets']))
df['lnOtherCurrLiabil'] = pd.DataFrame(np.log(df['Other Current Liabilities']))
df['lnOtherEquity'] = pd.DataFrame(np.log(df['Other Equity']))
df['lnOtherFinancingActivit'] = pd.DataFrame(np.log(df['Other Financing Activities']))
df['lnOtherInvestingActivit'] = pd.DataFrame(np.log(df['Other Investing Activities']))
df['lnOtherLiabilities'] = pd.DataFrame(np.log(df['Other Liabilities']))
df['lnOtherOperatingActivit'] = pd.DataFrame(np.log(df['Other Operating Activities']))
df['lnOtherOperatingItems'] = pd.DataFrame(np.log(df['Other Operating Items']))
df['lnPreTaxMargin'] = pd.DataFrame(np.log(df['Pre-Tax Margin']))
df['lnPreTaxROE'] = pd.DataFrame(np.log(df['Pre-Tax ROE']))
df['lnProfitMargin'] = pd.DataFrame(np.log(df['Profit Margin']))
df['lnQuickRatio'] = pd.DataFrame(np.log(df['Quick Ratio']))
df['lnRnD'] = pd.DataFrame(np.log(df['Research and Development']))
df['lnRetainedEarning'] = pd.DataFrame(np.log(df['Retained Earnings']))
df['lnSalePurchasStock'] = pd.DataFrame(np.log(df['Sale and Purchase of Stock']))
df['lnSalesGeneralAdmin'] = pd.DataFrame(np.log(df['Sales, General and Admin.']))
df['lnSTermDebtTOCurrPortionLTermDebt'] = pd.DataFrame(np.log(df['Short-Term Debt / Current Portion of Long-Term Debt']))
df['lnSTermInvest'] = pd.DataFrame(np.log(df['Short-Term Investments']))
df['lnTotalAssets'] = pd.DataFrame(np.log(df['Total Assets']))
df['lnTotalCurrAssets'] = pd.DataFrame(np.log(df['Total Current Assets']))
df['lnTotalCurrLiabili'] = pd.DataFrame(np.log(df['Total Current Liabilities']))
df['lnTotalEquity'] = pd.DataFrame(np.log(df['Total Equity']))
df['lnTotalLiabilities'] = pd.DataFrame(np.log(df['Total Liabilities']))
df['lnTotalLiabilitiesEquity'] = pd.DataFrame(np.log(df['Total Liabilities & Equity']))
df['lnTotalRevenue'] = pd.DataFrame(np.log(df['Total Revenue']))
df['lnTreasuryStock'] = pd.DataFrame(np.log(df['Treasury Stock']))
df['lnEPS'] = pd.DataFrame(np.log(df['Earnings Per Share']))
df['lnEstSharesOutstdg'] = pd.DataFrame(np.log(df['Estimated Shares Outstanding']))

#Dropping original large scale variables
df = df.drop(['Accounts Payable',
 'Accounts Receivable',
 "Add'l income/expense items",
 'After Tax ROE',
 'Capital Expenditures',
 'Capital Surplus',
 'Cash Ratio',
 'Cash and Cash Equivalents',
 'Changes in Inventories',
 'Common Stocks',
 'Cost of Revenue',
 'Current Ratio',
 'Deferred Asset Charges',
 'Deferred Liability Charges',
 'Depreciation',
 'Earnings Before Interest and Tax',
 'Earnings Before Tax',
 'Effect of Exchange Rate',
 'Equity Earnings/Loss Unconsolidated Subsidiary',
 'Fixed Assets',
 'Goodwill',
 'Gross Margin',
 'Gross Profit',
 'Income Tax',
 'Intangible Assets',
 'Interest Expense',
 'Inventory',
 'Investments',
 'Liabilities',
 'Long-Term Debt',
 'Long-Term Investments',
 'Minority Interest',
 'Misc. Stocks',
 'Net Borrowings',
 'Net Cash Flow',
 'Net Cash Flow-Operating',
 'Net Cash Flows-Financing',
 'Net Cash Flows-Investing',
 'Net Income',
 'Net Income Adjustments',
 'Net Income Applicable to Common Shareholders',
 'Net Income-Cont. Operations',
 'Net Receivables',
 'Non-Recurring Items',
 'Operating Income',
 'Operating Margin',
 'Other Assets',
 'Other Current Assets',
 'Other Current Liabilities',
 'Other Equity',
 'Other Financing Activities',
 'Other Investing Activities',
 'Other Liabilities',
 'Other Operating Activities',
 'Other Operating Items',
 'Pre-Tax Margin',
 'Pre-Tax ROE',
 'Profit Margin',
 'Quick Ratio',
 'Research and Development',
 'Retained Earnings',
 'Sale and Purchase of Stock',
 'Sales, General and Admin.',
 'Short-Term Debt / Current Portion of Long-Term Debt',
 'Short-Term Investments',
 'Total Assets',
 'Total Current Assets',
 'Total Current Liabilities',
 'Total Equity',
 'Total Liabilities',
 'Total Liabilities & Equity',
 'Total Revenue',
 'Treasury Stock',
 'Earnings Per Share',
 'Estimated Shares Outstanding',
 'Unnamed: 0',
 'For Year',
 'Period Ending',
 'Mean_Return'], axis = 1)

#==Testing for MultiCollinearity
#Variance Inflation Factors:
#Code for VIF Calculation
#a function to calculate the VIF values

def VIF_cal(mcl):
    import statsmodels.formula.api as smf
    x_vars = mcl
    xvar_names = x_vars.columns
    for i in range(0,len(xvar_names)):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),3)        
        print(xvar_names[i], "VIF = ", vif)

#Calculating VIF values using the VIF_cal function and drop varibles with vif>4
mcl = df.drop(['Name', 'Return_cat', 'Adj_Date', 'lnNetIncome', 'lnTotalAssets', 
               'lnCurRatio', 'lnOperIncome', 'lnNetIncomeApplicableCmonShrholdrs', 'lnEBT', 'lnEBIT',
               'lnNetCashFlow', 'lnPreTaxROE', 'lnTotalLiabilitiesEquity', 'lnTotalCurrLiabili',
               'lnTotalRevenue','lnTotalCurrAssets', 'lnNetCFOperating', 'lnCashRatio', 'lnTotalLiabilities', 'lnPreTaxMargin',
               'lnDepreciation', 'lnEPS', 'lnOperMargin','lnAccPay', 'lnNetCFInvesting', 'lnNetCFFinancing', 'lnFixedAssets', 'lnLiabilities', 'lnQuickRatio', 'lnEstSharesOutstdg'], axis = 1)
VIF_cal(mcl)
mcl.shape

mcl.notnull().sum()

#==========
#==== extracting independent and target variables
X = mcl.copy()
y = df['Return_cat']

#=========
#====Feature Selection
#Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Feature extraction
logireg = LogisticRegression()
rfe = RFE(logireg, 10)
fit = rfe.fit(X, y)
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

list(X)

X = X.iloc[:,[1,2,3,5,10,15,20,30,33,46]]
y = df['Return_cat']


#=========
#==== Spliting dataset into traing  and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0) 


# Fitting Multiclass Logistic Classification to the Training set
logireg = LogisticRegression()
#==== Train Regressor
logireg.fit(X_train, y_train)

#==== Predict on the test set
y_pred_logireg = logireg.predict(X_test)


#==== Performance Measures
logireg.score(X_test,y_test)

#lets see the actual and predicted value side by side
y_compare = pd.DataFrame(np.vstack((y_test,y_pred_logireg)).T, columns=['y_test','y_pred_logireg'])


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test,y_pred_logireg)

confusion_matrix(y_test,y_pred_logireg)

cr_logireg = classification_report(y_test,y_pred_logireg)


#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score #, cross_val_predict
#from sklearn import metrics
scores_logireg = cross_val_score(logireg.fit(X_train,y_train), X_train, y_train, cv=8)
print ('Cross-validated scores (cv=6) logreg:', scores_logireg)






#############OTHER MODELS
#############OTHER MODELS
#############OTHER MODELS
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.svm  import SVC
from sklearn.ensemble  import RandomForestClassifier, AdaBoostClassifier

knn = KNeighborsClassifier(n_neighbors=7, weights='distance') 
rfc = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0) 
adbc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0) 
svmc = SVC(kernel='poly', degree=2, gamma='scale') 


#==== Train Regressor
knn.fit(X_train,y_train)
rfc.fit(X_train,y_train)
adbc.fit(X_train,y_train)
svmc.fit(X_train,y_train)

#==== Predict on the test set
y_pred_knn = knn.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_adbc = adbc.predict(X_test)
y_pred_svmc = svmc.predict(X_test)

#==== Performance Measures
logireg.score(X_test,y_test)
knn.score(X_test,y_test)
rfc.score(X_test,y_test)
adbc.score(X_test,y_test)
svmc.score(X_test,y_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_test,y_pred_logireg)
accuracy_score(y_test,y_pred_knn)
accuracy_score(y_test,y_pred_rfc)
accuracy_score(y_test,y_pred_adbc)
accuracy_score(y_test,y_pred_svmc)


confusion_matrix(y_test,y_pred_logireg)
confusion_matrix(y_test,y_pred_knn)
confusion_matrix(y_test,y_pred_rfc)
confusion_matrix(y_test,y_pred_adbc)
confusion_matrix(y_test,y_pred_svmc)

cr_logireg = classification_report(y_test,y_pred_logireg)
cr_knn = classification_report(y_test,y_pred_knn)
cr_rfc = classification_report(y_test,y_pred_rfc)
cr_adbc = classification_report(y_test,y_pred_adbc)
cr_svmc = classification_report(y_test,y_pred_svmc)


#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores_logireg = cross_val_score(logireg.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) logreg:', scores_logireg)

scores_knn = cross_val_score(knn.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) knn:', scores_knn)

scores_rfc = cross_val_score(rfc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) rfc:', scores_rfc)

scores_adbc = cross_val_score(adbc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) adbc:', scores_adbc)

scores_svmc = cross_val_score(svmc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) svmc:', scores_svmc)
 

#====
#====
#==== Grid Search hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV
# Linear Regression does not need hyper-parameter tuning
# KNN
model_knn1 = KNeighborsClassifier() 

param_dict_knn = {
        'n_neighbors': [5,6,7,9,11], 
        'weights': ['uniform', 'distance'], 
        'leaf_size' : [10,20,25,30,35,40],
        }

model_knn2 = GridSearchCV(model_knn1,param_dict_knn)
model_knn2.fit(X_train,y_train)
model_knn2.best_params_
model_knn2.best_score_

# Random Forest Regressor
model_rfc1 = RandomForestClassifier() 

param_dict_rfc = {
        'n_estimators': [20,30,40,50,60], 
        'max_depth': [10,20,30,40,50],         
        }

model_rfc2 = GridSearchCV(model_rfc1, param_dict_rfc)
model_rfc2.fit(X_train,y_train)
model_rfc2.best_params_
model_rfc2.best_score_

# AdaBoost Regressor
model_adbc1 = AdaBoostClassifier()

param_dict_adbc = {
        'n_estimators': [30,40,50,60,70],        
        'learning_rate' : [.1,1,3,10],
        }

model_adbc2 = GridSearchCV(model_adbc1, param_dict_adbc)
model_adbc2.fit(X_train,y_train)
model_adbc2.best_params_
model_adbc2.best_score_

# SVC
model_svmc1 = SVC()

param_dict_svmc = {
        'gamma': ['auto', 'scale'],
        'C' : [0.001,0.01,0.1,1,10],
        'kernel' : ['rbf', 'linear','poly', 'sigmoid'],        
        'degree' : [1,2,3,4,5]
        }

model_svmc2 = GridSearchCV(model_svmc1, param_dict_svmc, cv=None)
model_svmc2.fit(X_train,y_train)
model_svmc2.best_params_
model_svmc2.best_score_


#Comparison between Initial classifiers and the best param_s on TRAIN SET
model_knn2.best_score_
model_rfc2.best_score_
model_adbc2.best_score_
model_svmc2.best_score_

logireg.score(X_test,y_test)
knn.score(X_train,y_train)
rfc.score(X_train,y_train)
adbc.score(X_train,y_train)
svmc.score(X_train,y_train)

#Comparison between Initial classifiers and the best param_s on TEST SET
knn3 = KNeighborsClassifier(leaf_size=10, n_neighbors=11, weights='distance') 
rfc3 = RandomForestClassifier(n_estimators=60, max_depth=20, random_state=0) 
adbc3 = AdaBoostClassifier(n_estimators=30, learning_rate=0.1, random_state=0) 
svmc3 = SVC(kernel='poly', degree=2, C=10, gamma='auto') 

#==== Train Regressor
knn3.fit(X_train,y_train)
rfc3.fit(X_train,y_train)
adbc3.fit(X_train,y_train)
svmc3.fit(X_train,y_train)

'''
#==== Predict on the test set
y_pred_knn3 = knn3.predict(X_test)
y_pred_rfc3 = rfc3.predict(X_test)
y_pred_adbc3 = adbc3.predict(X_test)
y_pred_svmc3 = svmc3.predict(X_test)
'''

#==== Performance Measures
logireg.score(X_test,y_test)
knn3.score(X_test,y_test)
rfc3.score(X_test,y_test)
adbc3.score(X_test,y_test)
svmc3.score(X_test,y_test)


logireg.score(X_test,y_test)
knn.score(X_test,y_test)
rfc.score(X_test,y_test)
adbc.score(X_test,y_test)
svmc.score(X_test,y_test)