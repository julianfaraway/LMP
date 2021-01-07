# # Problems with the Error
# ## Generalized Least Squares
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import faraway.utils


#	

import faraway.datasets.globwarm
globwarm = faraway.datasets.globwarm.load()
lmod = smf.ols('nhtemp ~ wusa + jasper + westgreen + chesapeake +\
    tornetrask +  urals + mongolia + tasman', globwarm).fit()
lmod.sumary()


#	

np.corrcoef(lmod.resid.iloc[:-1],lmod.resid.iloc[1:]).round(3)


#	

lmod.resid.autocorr()


#	

X = lmod.model.wexog
y = lmod.model.wendog
gmod = sm.GLSAR(y, X, rho=1)
res=gmod.iterative_fit(maxiter=6)
gmod.rho.round(3)


#	

res.summary().tables[1]


# ## Weighted Least Squares
#	

import faraway.datasets.fpe
fpe = faraway.datasets.fpe.load()
fpe.head()


#	

lmod = smf.wls("A2 ~ A + B + C + D + E + F + G + H + J + K +N -1",
    fpe).fit()


#	

wmod = smf.wls("A2 ~ A + B + C + D + E + F + G + H + J + K +N -1",
    fpe, weights = 1/fpe.EI ).fit()


#	

wmod53 = smf.wls("A2 ~ A + B + C + D + E + F + G + H + J + K+N-1",
    fpe, weights = 53/fpe.EI ).fit()


#	

pd.DataFrame([lmod.params, wmod.params, wmod53.params],
    index=['no weights','weights','weights*53']).round(3)


#	

y = fpe.A2 - fpe.A - fpe.G - fpe.K
X = fpe.loc[:,["C","D","E","F","N"]]
wmod = sm.WLS(y, X, weights = 1/fpe.EI ).fit()
wmod.params


#	

y = fpe.A2
X = fpe.loc[:,["A","B","C","D","E","F","G","H","J","K","N"]]
weights = 1/fpe.EI
Xw = (X.T * np.sqrt(weights)).T
yw = y * np.sqrt(weights)


#	

res = sp.optimize.lsq_linear(Xw, yw, bounds=(0, 1))
pd.Series(np.round(res.x,3),index=lmod.params.index)


#	

import faraway.datasets.cars
cars = faraway.datasets.cars.load()
sns.regplot(x='speed', y = 'dist', data = cars)


#	

lmod = smf.ols('dist ~ speed', cars).fit()
g = sns.regplot(lmod.fittedvalues, np.sqrt(abs(lmod.resid)), 
    lowess = True)
g.set_xlabel("Fitted Values")
g.set_ylabel("Log-squared Residuals")


#	

x = lmod.fittedvalues
y = np.log(lmod.resid**2)
z = sm.nonparametric.lowess(y,x)
w = 1.0/np.exp(z[:,1])
wmod = smf.wls('dist ~ speed', cars, weights =w).fit()
wmod.summary().tables[1]


#	

print(lmod.summary().tables[1])


# ## Testing for Lack of Fit
#	

import faraway.datasets.corrosion
corrosion = faraway.datasets.corrosion.load()
lmod = smf.ols('loss ~ Fe', corrosion).fit()
lmod.sumary()


#	

corrosion['Fefac'] = corrosion['Fe'].astype('category')
amod = smf.ols('loss ~ Fefac', corrosion).fit()


#	

fig, ax = plt.subplots()
ax.scatter(corrosion.Fe, corrosion.loss)
plt.xlabel("Fe")
plt.ylabel("loss")
xr = np.array(ax.get_xlim())
ax.plot(xr, lmod.params[0] + lmod.params[1] * xr)
ax.scatter(corrosion.Fe, amod.fittedvalues, marker='x',s=100)


#	

amod.compare_f_test(lmod)


#	

pc = np.polyfit(corrosion.Fe, corrosion.loss, 6)


#	

fig, ax = plt.subplots()
ax.scatter(corrosion.Fe, corrosion.loss)
plt.xlabel("Fe")
plt.ylabel("loss")
ax.scatter(corrosion.Fe, amod.fittedvalues, marker='x')
grid = np.linspace(0,2,50)
ax.plot(grid,np.poly1d(pc)(grid))


#	

pc, rss, _, _, _ = np.polyfit(corrosion.Fe, corrosion.loss, 
    6, full=True)
tss = np.sum((corrosion.loss-np.mean(corrosion.loss))**2)
1-rss/tss


# ## Robust Regression
# ### M-Estimation
#	

import faraway.datasets.galapagos
galapagos = faraway.datasets.galapagos.load()
lsmod = smf.ols(
    'Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', 
    galapagos).fit()
lsmod.sumary()


#	

X = lsmod.model.wexog
y = lsmod.model.wendog
rlmod = sm.RLM(y,X).fit()
rlmod.summary()


#	

wts = rlmod.weights
wts[wts < 1]


#	

l1mod = sm.QuantReg(y,X).fit()
l1mod.summary()


# ### High Breakdown Estimators
#	

import faraway.datasets.star
star = faraway.datasets.star.load()
gs1 = smf.ols('light ~ temp', star).fit()
X = gs1.model.wexog
gs2 = sm.RLM(star.light, X, data=star).fit()
gs3 = smf.ols('light ~ temp', star.loc[star.temp > 3.6,:]).fit()
plt.scatter(star.temp, star.light, label = None)
xr = np.array([min(star.temp), max(star.temp)])
plt.plot(xr, gs1.params[0] + gs1.params[1]*xr,'k-',label="OLS")
plt.plot(xr, gs2.params[0] + gs2.params[1]*xr,'k--',label="Huber")
plt.plot(xr, gs3.params[0] + gs3.params[1]*xr,'k:',label="OLS-4")
plt.legend()


#	

from sklearn.linear_model import TheilSenRegressor
X = np.reshape(star.temp.array,(-1,1))
y = star.light
reg = TheilSenRegressor(random_state=0).fit(X,y)
plt.scatter(star.temp, star.light)
plt.plot(xr, reg.intercept_ + reg.coef_*xr,'k-')


#	

from sklearn.linear_model import  RANSACRegressor
reg = RANSACRegressor().fit(X,y)
i = reg.inlier_mask_
plt.scatter(star.temp[i], star.light[i])
plt.scatter(star.temp[~i], star.light[~i],marker='x')
plt.plot(xr, reg.estimator_.intercept_ + reg.estimator_.coef_*xr,
    'k-')


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

    