# # Shrinkage Methods
# ## Principal Components
#	

import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils
import matplotlib.pyplot as plt


#	

import faraway.datasets.fat
fat = faraway.datasets.fat.load()


#	

plt.scatter(fat.knee, fat.neck)
plt.xlabel("Knee")
plt.ylabel("Neck")


#	

plt.scatter(fat.thigh, fat.chest)
plt.xlabel("Thigh")
plt.ylabel("Chest")


#	

plt.scatter(fat.wrist, fat.hip)
plt.xlabel("Wrist")
plt.ylabel("Hip")


#	

from sklearn.decomposition import PCA
pca = PCA()
cfat = fat.iloc[:,8:]
pca.fit(cfat)


#	

np.sqrt(pca.explained_variance_).round(2)


#	

pca.explained_variance_ratio_.round(3)


#	

rot = pca.components_
pd.DataFrame(rot[0,:],index=cfat.columns).round(3).T


#	

from sklearn.preprocessing import scale
scalfat = pd.DataFrame(scale(cfat))
pcac = PCA()
pcac.fit(scalfat)
pcac.explained_variance_ratio_.round(2)


#	

rot = pcac.components_
pd.DataFrame(rot[0,:],index=cfat.columns).round(3).T


#	

pd.DataFrame(rot[1,:],index=cfat.columns).round(3).T


#	

from sklearn.covariance import EllipticEnvelope
ee = EllipticEnvelope()
ee.fit(cfat)


#	

md = np.sqrt(ee.mahalanobis(cfat))
n=len(md)
ix = np.arange(1,n+1)
halfq = sp.stats.norm.ppf((n+ix)/(2*n+1)),
plt.scatter(halfq, np.sort(md))
plt.xlabel(r'$\chi^2$ quantiles')
plt.ylabel('Mahalanobis distances')


#	

xmat = sm.add_constant(cfat)
lmod = sm.OLS(fat.brozek, xmat).fit()
lmod.sumary()


#	

pcscores = pca.fit_transform(scale(cfat))
xmat = sm.add_constant(pcscores[:,:2])
lmod = sm.OLS(fat.brozek, xmat).fit()
lmod.sumary()


#	

xmat = pd.concat([scalfat.iloc[:,2], 
    scalfat.iloc[:,6] - scalfat.iloc[:,2]],axis=1)
xmat.columns = ['overall','muscle']
xmat = sm.add_constant(xmat)
lmod = sm.OLS(fat.brozek, xmat).fit()
lmod.sumary()


#	

import faraway.datasets.meatspec
meatspec = faraway.datasets.meatspec.load()
trainmeat = meatspec.iloc[:172,]
testmeat = meatspec.iloc[173:,]


#	

from sklearn import linear_model
fullreg = linear_model.LinearRegression(fit_intercept=True)
Xtrain = trainmeat.drop('fat',axis=1)
fullreg.fit(Xtrain, trainmeat.fat)
fullreg.score(Xtrain, trainmeat.fat)


#	

from sklearn.metrics import mean_squared_error


#	

np.sqrt(mean_squared_error(fullreg.predict(Xtrain),trainmeat.fat))


#	

Xtest = testmeat.drop('fat',axis=1)
np.sqrt(mean_squared_error(fullreg.predict(Xtest),testmeat.fat))


#	

frequency = np.arange(0,100)
plt.plot(frequency,Xtrain.T,alpha=0.15)
plt.xlabel("Frequency")


#	

from sklearn.feature_selection import RFECV
redreg = linear_model.LinearRegression(fit_intercept=True)
redreg.fit(Xtrain, trainmeat.fat)
selector = RFECV(redreg, step=1, cv=10)
selector = selector.fit(Xtrain, trainmeat.fat)
Xred = Xtrain.iloc[:,selector.support_]
Xred.shape


#	

redreg.fit(Xred,trainmeat.fat)
np.sqrt(mean_squared_error(redreg.predict(Xred),trainmeat.fat))


#	

Xredtest = Xtest.iloc[:,selector.support_]
np.sqrt(mean_squared_error(redreg.predict(Xredtest),testmeat.fat))


#	

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(Xtrain)
np.sqrt(pca.explained_variance_).round(2)[:10]


#	

plt.plot(frequency, pca.components_[0,:],'k-',label="PC1")
plt.plot(frequency, pca.components_[1,:],'k:',label="PC2")
plt.plot(frequency, pca.components_[2,:],'k--',label="PC3")
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


#	

pcscores = pca.fit_transform(Xtrain)
pc4reg = linear_model.LinearRegression(fit_intercept=True)
pc4 = pcscores[:,:4]
pc4reg.fit(pc4,trainmeat.fat)
np.sqrt(mean_squared_error(pc4reg.predict(pc4),trainmeat.fat))


#	

plt.plot(frequency, fullreg.coef_)
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


#	

pceff = np.dot(pca.components_[:4,].T, pc4reg.coef_)
plt.plot(frequency, pceff)
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


#	

rotX = (Xtest - pca.mean_) @ pca.components_[:4,].T
np.sqrt(mean_squared_error(pc4reg.predict(rotX),testmeat.fat))


#	

maxcomp = 50
ncomp = np.arange(1,maxcomp+1)
rmsep = np.empty(maxcomp)
pcrmod = linear_model.LinearRegression(fit_intercept=True)
for icomp in ncomp:
    rotX =  (Xtest - pca.mean_) @ pca.components_[:icomp,].T
    pcsi = pcscores[:,:icomp]
    pcrmod.fit(pcsi,trainmeat.fat)
    rmsep[icomp-1] = np.sqrt(mean_squared_error(
        pcrmod.predict(rotX),testmeat.fat))
plt.plot(ncomp, rmsep)
plt.ylabel("RMSE")
plt.xlabel("Number of components")


#	

np.argmin(rmsep)+1, min(rmsep)


#	

from sklearn.model_selection import cross_val_predict
rmsepcv = np.empty(len(ncomp))
np.random.seed(123)
for i in ncomp:
    pcsi = pcscores[:,:i]
    pcrmod.fit(pcsi,trainmeat.fat)
    ypred = cross_val_predict(pcrmod, pcsi, trainmeat.fat, cv=10)
    rmsepcv[i-1] = np.sqrt(mean_squared_error(
                   ypred,trainmeat.fat))
plt.plot(ncomp,rmsepcv)
plt.ylabel("RMSE")
plt.xlabel("Number of components")


#	

np.argmin(rmsepcv)+1, min(rmsepcv)


#	

rmsep[np.argmin(rmsepcv)]


# ## Partial Least Squares
#	

from sklearn.cross_decomposition import PLSRegression
plsreg = PLSRegression(scale=False, n_components=4)
plsmod = plsreg.fit(Xtrain, trainmeat.fat)


#	

plt.plot(frequency,plsmod.x_loadings_[:,0],'k-',label="1")
plt.plot(frequency,plsmod.x_loadings_[:,1],'k:',label="2")
plt.plot(frequency,plsmod.x_loadings_[:,2],'k--',label="3")
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


#	

plt.plot(frequency, plsmod.coef_)
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


#	

np.sqrt(mean_squared_error(plsmod.predict(Xtrain),trainmeat.fat))


#	

ncomp = 50
rmsep = np.empty(ncomp)
component = np.arange(1, ncomp)
np.random.seed(123)
for i in range(ncomp):
    pls = PLSRegression(scale=False,n_components=i+1)
    ypred = cross_val_predict(pls, Xtrain, trainmeat.fat, cv=10)
    rmsep[i] = np.sqrt(mean_squared_error(ypred,trainmeat.fat))
plt.plot(range(1,ncomp+1),rmsep)
plt.ylabel("RMSE")
plt.xlabel("Number of components")


#	

np.argmin(rmsep)+1


#	

plsbest = PLSRegression(scale=False,n_components=14)
plsbest.fit(Xtrain, trainmeat.fat)
ypred = plsbest.predict(Xtrain)
np.sqrt(mean_squared_error(ypred,trainmeat.fat))


#	

ytpred = plsbest.predict(Xtest)
np.sqrt(mean_squared_error(ytpred,testmeat.fat))


# ## Ridge Regression
#	

n_alphas = 50
alphas = np.logspace(-10, -5, n_alphas)


#	

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a)
    ridge.fit(Xtrain, trainmeat.fat)
    coefs.append(ridge.coef_)


#	

ax = plt.gca()
ax.plot(alphas, coefs, 'k', alpha=0.2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('alpha')
plt.ylabel('Coefficients')


#	

from sklearn.model_selection import GridSearchCV
pars = {'alpha':np.logspace(-10, -5, 50)}
rr= GridSearchCV(linear_model.Ridge(), pars, scoring='r2', cv=10)
rr.fit(Xtrain, trainmeat.fat)


#	

bestalpha = rr.best_params_['alpha']
bestalpha


#	

rrbest = linear_model.Ridge(alpha=bestalpha, fit_intercept=True)
rrbest.fit(Xtrain, trainmeat.fat)
np.sqrt(mean_squared_error(rrbest.predict(Xtrain),trainmeat.fat))


#	

np.sqrt(mean_squared_error(rrbest.predict(Xtest),testmeat.fat))


#	

plt.plot(frequency,rrbest.coef_)
plt.xlabel("Frequency")
plt.ylabel("Coefficient")


# ## Lasso
#	

import faraway.datasets.fat
fat = faraway.datasets.fat.load()
X = fat.iloc[:,8:]


#	

n_alphas = 50
alphas = np.logspace(-2, 2, n_alphas)


#	

lasso = linear_model.Lasso()
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X, fat.brozek)
    coefs.append(lasso.coef_)


#	

lassocv = linear_model.LassoCV(cv = 10)
lassocv.fit(X, fat.brozek)
lassocv.alpha_


#	

min(lassocv.alphas_),max(lassocv.alphas_)


#	

ax = plt.gca()
ax.plot(alphas, coefs)
plt.xlim(min(alphas)/5,max(alphas))
ax.set_xscale('log')
for i in range(len(X.columns)):
    plt.text(min(alphas)/4,coefs[0][i],X.columns[i])
plt.xlabel('alpha')
plt.ylabel('coefficients')
plt.axvline(lassocv.alpha_)


#	

lassocv.coef_.round(3)


#	

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, fat.brozek)
reg.coef_.round(3)


#	

lasso = linear_model.Lasso(max_iter=1e6,tol=0.001)
lasso.set_params(alpha=1e-9)
lasso.fit(Xtrain, trainmeat.fat)


#	

np.mean(abs(lasso.coef_)>1e-5)


#	

np.sqrt(mean_squared_error(lasso.predict(Xtrain),trainmeat.fat))


#	

np.sqrt(mean_squared_error(lasso.predict(Xtest),testmeat.fat))


# ## Other Methods
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

    