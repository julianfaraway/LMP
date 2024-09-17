# # Estimation
# ## Linear Model
# ## Matrix Representation
# ## Estimating $\beta$
# ## Least Squares Estimation
# ## Examples of Calculating $\hat\beta$
# ## Example
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns


#	

import faraway.datasets.galapagos
galapagos = faraway.datasets.galapagos.load()
galapagos.head()


#	

lmod = smf.ols(
    formula='Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', 
    data=galapagos).fit()
lmod.summary()


#	

import faraway.utils
lmod.sumary()


#	

X = galapagos.iloc[:,1:]
X.insert(0,'intercept',1)


#	

XtXi = np.linalg.inv(X.T @ X)


#	

(XtXi @ X.T) @ galapagos.Species


#	

np.linalg.solve(X.T @ X, X.T @ galapagos.Species)


#	

np.sqrt(lmod.mse_resid)


# ## Computing Least Squares Estimates
#	

Xmp = np.linalg.pinv(X)
Xmp.shape


#	

Xmp @ galapagos.Species


#	

q, r = np.linalg.qr(X)


#	

f = q.T @ galapagos.Species
f


#	

sp.linalg.solve_triangular(r, f)


#	

lmodform = smf.ols(
    'Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', 
    galapagos)
lmod = lmodform.fit(method="qr")
lmod.params


#	

params, res, rnk, s = sp.linalg.lstsq(X, galapagos['Species'])
params


# ## Gauss--Markov Theorem
# ## Goodness of Fit
#	

x = np.linspace(0,1,101)
np.random.seed(123)
y = x + np.random.normal(0,0.1,101)
plt.scatter(x,y,alpha=0.75)
plt.xlabel("x")
plt.ylabel("y")
beta1, beta0 = np.polyfit(x,y,1)
plt.plot([0,1],[beta0,beta0+beta1],"k-")
plt.annotate("",xy=(0.2, min(y)),xytext=(0.2, max(y)),
    arrowprops=dict(arrowstyle="<->",lw=2))
plt.annotate("",xy=(0.5, 0.3),   xytext=(0.5, 0.7),
    arrowprops=dict(arrowstyle="<->",linestyle="--",lw=2))


#	

df = sns.load_dataset("anscombe")
sns.lmplot(x="x", y="y", col="dataset", 
    data=df, col_wrap=2, ci=None)


#	

df.groupby("dataset").corr().iloc[0::2,-1]**2


# ## Identifiability
#	

galapagos['Adiff'] = galapagos['Area'] - galapagos['Adjacent']


#	

lmodform = smf.ols(
   'Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiff', 
   galapagos)
lmod = lmodform.fit()
lmod.sumary()


#	

lmod.eigenvals[-1]


#	

lmod = lmodform.fit(method="qr")
lmod.sumary()


#	

np.random.seed(123)
galapagos['Adiffe'] = galapagos['Adiff'] + \
   (np.random.rand(30)-0.5)*0.001 


#	

lmod = smf.ols(
    'Species ~ Area+Elevation+Nearest+Scruz+Adjacent+Adiffe', 
    galapagos).fit()
lmod.sumary()


# ## Orthogonality
#	

import faraway.datasets.odor
odor = faraway.datasets.odor.load()


#	

odor.iloc[:,1:].cov()


#	

lmod = smf.ols('odor ~ temp + gas + pack', odor).fit()
lmod.params


#	

lmod.cov_params()


#	

lmod = smf.ols('odor ~ gas + pack', odor).fit()
lmod.params


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

    
