# # Model Selection
# ## Hierarchical Models
# ## Hypothesis Testing-Based Procedures
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils


#	

import faraway.datasets.statedata
statedata = faraway.datasets.statedata.load()
statedata.index = statedata['State']
statedata = statedata.drop('State',1)
statedata.head()


#	

lmod = smf.ols('LifeExp ~ Population + Income + Illiteracy + \
    Murder + HSGrad + Frost + Area', statedata).fit()
lmod.sumary()


#	

lmod.pvalues.idxmax(), lmod.pvalues.max()


#	

lmod = smf.ols('LifeExp ~ Population + Income + Illiteracy + \
    Murder + HSGrad + Frost', statedata).fit()
lmod.pvalues.idxmax(), lmod.pvalues.max()


#	

lmod = smf.ols(
    'LifeExp ~ Population + Income + Murder + HSGrad + Frost', 
    statedata).fit()
lmod.pvalues.idxmax(), lmod.pvalues.max()


#	

lmod = smf.ols(
    'LifeExp ~ Population + Murder + HSGrad + Frost', 
    statedata).fit()
lmod.pvalues.idxmax(), lmod.pvalues.max()


#	

lmod = smf.ols(
    'LifeExp ~ Murder + HSGrad + Frost', statedata).fit()
lmod.sumary()


#	

lmod = smf.ols(
    'LifeExp ~ Illiteracy + Murder + Frost', statedata).fit()
lmod.sumary()


# ## Criterion-Based Procedures
#	

import itertools
pcols = list(statedata.columns)
pcols.remove('LifeExp')
rss = np.empty(len(pcols) + 1)
rss[0] = np.sum(
    (statedata.LifeExp - np.mean(statedata.LifeExp))**2)
selvar = ['Null']
for k in range(1,len(pcols)+1):
    RSS = {}
    for variables in itertools.combinations(pcols, k):
        predictors = statedata.loc[:,list(variables)]
        predictors['Intercept'] = 1
        res = sm.OLS(statedata.LifeExp, predictors).fit()
        RSS[variables] = res.ssr
    rss[k] = min(RSS.values())
    selvar.append(min(RSS, key=RSS.get))
rss.round(3)


#	

['Null',
 ('Murder',),
 ('Murder', 'HSGrad'),
 ('Murder', 'HSGrad', 'Frost'),
 ('Population', 'Murder', 'HSGrad', 'Frost'),
 ('Population', 'Income', 'Murder', 'HSGrad', 'Frost'),
 ('Population', 'Income', 'Illiteracy', 'Murder', 'HSGrad', 
    'Frost'),
 ('Population', 'Income', 'Illiteracy', 'Murder', 'HSGrad', 
    'Frost', 'Area')]


#	

aic = 50 * np.log(rss/50) + np.arange(1,9)*2
plt.plot(np.arange(1,8), aic[1:])
plt.xlabel('Number of Predictors')
plt.ylabel('AIC')


#	

adjr2 = 1 - (rss/(50 - np.arange(1,9)))/(rss[0]/49)
plt.plot(np.arange(1,8),adjr2[1:])
plt.xlabel('Number of Predictors')
plt.ylabel('Adjusted R^2')


#	

from sklearn.preprocessing import scale
scalstat = pd.DataFrame(scale(statedata), index=statedata.index, 
    columns=statedata.columns)


#	

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=False)
X = scalstat.drop('LifeExp',axis=1)
reg.fit(X, scalstat.LifeExp)
reg.coef_


#	

reg.intercept_


#	

from sklearn.feature_selection import RFE
selector = RFE(reg, n_features_to_select=1)
selector = selector.fit(X, scalstat.LifeExp)
selector.ranking_


#	

X.columns[np.argsort(selector.ranking_)].tolist()


# ## Sample Splitting
#	

import faraway.datasets.fat
fat = faraway.datasets.fat.load()


#	

n = len(fat)
np.random.seed(123)
ii = np.random.choice(n,n//3,replace=False)
testfat = fat.iloc[ii]
trainfat = fat.drop(ii)


#	

from sklearn.preprocessing import scale
scalfat = pd.DataFrame(scale(trainfat), columns=fat.columns)


#	

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=False)
prednames = ['age','weight','height','neck','chest','abdom','hip',
    'thigh','knee','ankle','biceps','forearm','wrist']
X = scalfat.loc[:,prednames]
reg.fit(X, scalfat.brozek)


#	

from sklearn.feature_selection import RFE
selector = RFE(reg, n_features_to_select=1)
selector = selector.fit(X, scalfat.brozek)
selector.ranking_


#	

np.array(prednames)[np.argsort(selector.ranking_)]


#	

def rmse(x,y): return(np.sqrt(np.mean((x-y)**2)))


#	

Xtrain = trainfat.loc[:,prednames]
Xtest = testfat.loc[:,prednames]
pcols = Xtrain.shape[1]
prefs = np.argsort(selector.ranking_)
testpred = np.empty(pcols + 1)
testpred[0] = rmse(testfat.brozek, np.mean(trainfat.brozek))
trainpred = np.empty(pcols + 1)
trainpred[0] = rmse(trainfat.brozek, np.mean(trainfat.brozek))


#	

for k in range(1,pcols+1):
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(Xtrain.iloc[:,prefs[0:k]], trainfat.brozek)
    ypred = reg.predict(Xtest.iloc[:,prefs[0:k]])
    testpred[k] = rmse(ypred,testfat.brozek)
    ypred = reg.predict(Xtrain.iloc[:,prefs[0:k]])
    trainpred[k] = rmse(ypred,trainfat.brozek)


#	

testpred.round(3)


#	

plt.plot(testpred,"k-",label="test")
plt.plot(trainpred,"k:",label="train")
plt.xlabel("No of predictors")
plt.ylabel("RMSE")
plt.legend()


# ## Crossvalidation
#	

scalfat = pd.DataFrame(scale(fat), columns=fat.columns)
X = scalfat.loc[:,prednames]
reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(X, scalfat.brozek)


#	

from sklearn.feature_selection import RFECV
selector = RFECV(reg, step=1, cv=10,
     scoring='neg_mean_squared_error')
selector = selector.fit(X, scalfat.brozek)


#	

plt.plot(np.arange(1,14),-selector.grid_scores_)
plt.xlabel("No. of Predictors")
plt.ylabel("MSE")


#	

selector.ranking_


# ## Summary
# ## Exercises

# ## Packages Used

import sys
import matplotlib
import statsmodels as sm
import seaborn as sns
print("Python version:{}".format(sys.version))
print("matplotlib version: {}".format(matplotlib.__version__))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("statsmodels version: {}".format(sm.__version__))
print("seaborn version: {}".format(sns.__version__))

    