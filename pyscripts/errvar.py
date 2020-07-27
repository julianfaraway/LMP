# # Problems with the Predictors
# ## Errors in the Predictors
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils


#	

import faraway.datasets.cars
cars = faraway.datasets.cars.load()
est = np.polyfit(cars.speed, cars.dist, 1)
est.round(2)


#	

fig, ax = plt.subplots()
ax.scatter(cars.speed, cars.dist,label=None)
plt.xlabel("Speed")
plt.ylabel("Distance")
xr = np.array(ax.get_xlim())
np.random.seed(123)
ax.plot(xr, est[1] + est[0] * xr,label="0")
est1 = np.polyfit(cars.speed + np.random.normal(size=50), 
    cars.dist, 1)
ax.plot(xr, est1[1] + est1[0] * xr, 'k--',label="1")
est2 = np.polyfit(cars.speed + np.random.normal(scale=2,size=50), 
    cars.dist, 1)
ax.plot(xr, est2[1] + est2[0] * xr, 'k-.',label="2")
est5 = np.polyfit(cars.speed + np.random.normal(scale=5,size=50), 
    cars.dist, 1)
ax.plot(xr, est5[1] + est5[0] * xr, 'k:',label="5")
plt.legend(title='$\delta$')


#	

ee = pd.DataFrame.from_records([est, est1, est2, est5],
    columns=["slope","intercept"])
ee.insert(0,"SDdelta",[0,1,2,5])
print(ee.round(2).to_string(index=False))


#	

vv = np.repeat(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 
    [1000, 1000, 1000, 1000, 1000])
slopes = np.zeros(5000)
for i in range(5000):
    slopes[i] = np.polyfit(cars.speed+np.random.normal(
        scale=np.sqrt(vv[i]),size=50), cars.dist, 1)[0]


#	

betas = np.reshape(slopes, (5, 1000)).mean(axis=1)
betas = np.append(betas,est[0])
variances = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 0.5])
gv = np.polyfit(variances, betas,1)


#	

plt.scatter(variances, betas)
xr = np.array([0,1])
plt.plot(xr, np.array(gv[1] + gv[0]*xr))
plt.plot([0], [gv[1]], marker='x', markersize=6)


#	

gv.round(2)


# ## Changes of Scale
#	

import faraway.datasets.savings
savings = faraway.datasets.savings.load()
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings).fit()
lmod.sumary()


#	

lmod = smf.ols('sr ~ pop15 + pop75 + I(dpi/1000) + ddpi', 
    savings).fit()
lmod.sumary()


#	

scsav = savings.apply(sp.stats.zscore)
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', scsav).fit()
lmod.sumary()


#	

edf = pd.concat([lmod.params, lmod.conf_int()],axis=1).iloc[1:,]
edf.columns = ['estimate','lb','ub']
npreds = edf.shape[0]
fig, ax = plt.subplots()
ax.scatter(edf.estimate,np.arange(npreds))
for i in range(npreds):
    ax.plot([edf.lb[i], edf.ub[i]], [i, i])
ax.set_yticks(np.arange(npreds))
ax.set_yticklabels(edf.index)
ax.axvline(0)


#	

savings['age'] = np.where(savings.pop15 > 35, 0, 1)


#	

savings['dpis'] = sp.stats.zscore(savings.dpi)/2
savings['ddpis'] = sp.stats.zscore(savings.ddpi)/2
smf.ols('sr ~ age + dpis + ddpis', savings).fit().sumary()


# ## Collinearity
#	

import faraway.datasets.seatpos
seatpos = faraway.datasets.seatpos.load()
lmod = smf.ols(
    'hipcenter ~ Age+Weight+HtShoes+Ht+Seated+Arm+Thigh+Leg', 
    seatpos).fit()
lmod.sumary()


#	

seatpos.iloc[:,:-1].corr().round(3)


#	

X = lmod.model.wexog[:,1:]
XTX = X.T @ X
evals, evecs = np.linalg.eig(XTX)
evals = np.flip(np.sort(evals))
evals


#	

np.sqrt(evals[0]/evals[1:])


#	

from patsy import dmatrix
X = dmatrix("Age+Weight+HtShoes+Ht+Seated+Arm+Thigh+Leg", 
    seatpos, return_type='dataframe')
lmod = sm.OLS(X['Age'],X.drop('Age',axis=1)).fit()
lmod.rsquared, 1/(1-lmod.rsquared)


#	

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) \
    for i in range(X.shape[1])]
pd.Series(vif, X.columns)


#	

seatpos['hiperb'] = seatpos.hipcenter+ \
    np.random.normal(scale=10,size=38)
lmod = smf.ols(
    'hipcenter ~ Age+Weight+HtShoes+Ht+Seated+Arm+Thigh+Leg', 
    seatpos).fit()
lmodp = smf.ols(
    'hiperb ~ Age+Weight+HtShoes+Ht+Seated+Arm+Thigh+Leg', 
    seatpos).fit()
pd.DataFrame([lmod.params, lmodp.params],
    index=['original','perturbed']).round(3)


#	

lmod.rsquared, lmodp.rsquared


#	

pd.DataFrame.corr(X.iloc[3:,3:]).round(3)


#	

smf.ols('hipcenter ~ Age+Weight+Ht', seatpos).fit().sumary()


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

    