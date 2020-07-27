# # Transformation
# ## Transforming the Response
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils


#	

import faraway.datasets.savings
savings = faraway.datasets.savings.load()
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', 
    savings).fit()
X = lmod.model.wexog
n = savings.shape[0]
sumlogy = np.sum(np.log(savings.sr))
lam = np.linspace(0.5,1.5,100)
llk = np.empty(100)
for i in range(0, 100):
    lmod = sm.OLS(sp.stats.boxcox(savings.sr,lam[i]),X).fit()
    llk[i] = -(n/2)*np.log(lmod.ssr/n) + (lam[i]-1)*sumlogy
fig, ax = plt.subplots()
ax.plot(lam,llk)
ax.set_xlabel('$\lambda$')
ax.set_ylabel('log likelihood')
maxi = llk.argmax()
ax.vlines(lam[maxi],ymin=min(llk),ymax=max(llk),
    linestyle = 'dashed')
cicut = max(llk) - sp.stats.chi2.ppf(0.95,1)/2
rlam = lam[llk > cicut]
ax.hlines(cicut,xmin=rlam[0],xmax=rlam[-1],linestyle = 'dashed')
ax.vlines([rlam[0],rlam[-1]],ymin=min(llk),ymax=cicut,
    linestyle = 'dashed')


#	

import faraway.datasets.galapagos
galapagos = faraway.datasets.galapagos.load()
from patsy import dmatrix
X = dmatrix('Area + Elevation + Nearest + Scruz + Adjacent',
    data=galapagos)
n = galapagos.shape[0]
sumlogy = np.sum(np.log(galapagos.Species))
lam = np.linspace(-0.0,0.65,100)
llk = np.empty(100)
for i in range(0, 100):
    lmod = sm.OLS(sp.stats.boxcox(galapagos.Species,
                  lam[i]),X).fit()
    llk[i] = -(n/2)*np.log(lmod.ssr/n) + (lam[i]-1)*sumlogy
fig, ax = plt.subplots()
ax.plot(lam,llk)
ax.set_xlabel('$\lambda$')
ax.set_ylabel('log likelihood')
maxi = llk.argmax()
ax.vlines(lam[maxi],ymin=min(llk),ymax=max(llk),
    linestyle = 'dashed')
cicut = max(llk) - sp.stats.chi2.ppf(0.95,1)/2
rlam = lam[llk > cicut]
ax.hlines(cicut,xmin=rlam[0],xmax=rlam[-1],linestyle = 'dashed')
ax.vlines([rlam[0],rlam[-1]],ymin=min(llk),ymax=cicut,
    linestyle = 'dashed')


#	

import faraway.datasets.leafburn
leafburn = faraway.datasets.leafburn.load()
from patsy import dmatrix
X = dmatrix('nitrogen + chlorine + potassium',data=leafburn)
n = leafburn.shape[0]
alpha = np.linspace(-0.999,0,100)
llk = np.empty(100)
for i in range(0, 100):
    lmod = sm.OLS(np.log(leafburn.burntime+alpha[i]),X).fit()
    llk[i] = -(n/2)*np.log(lmod.ssr) - \
        np.sum(np.log(leafburn.burntime + alpha[i]))
fig, ax = plt.subplots()
ax.plot(alpha,llk)
ax.set_xlabel('$\alpha$')
ax.set_ylabel('log likelihood')
maxi = llk.argmax()
ax.vlines(alpha[maxi],ymin=min(llk),ymax=max(llk),
    linestyle = 'dashed')
cicut = max(llk) - sp.stats.chi2.ppf(0.95,1)/2
ralp = alpha[llk > cicut]
ax.hlines(cicut,xmin=ralp[0],xmax=ralp[-1],linestyle = 'dashed')
ax.vlines([ralp[0],ralp[-1]],ymin=min(llk),ymax=cicut,
    linestyle = 'dashed')    


# ## Transforming the Predictors
# ## Broken Stick Regression
#	

lmod1 = smf.ols('sr ~ pop15', savings[savings.pop15 < 35]).fit()
lmod2 = smf.ols('sr ~ pop15', savings[savings.pop15 > 35]).fit()
plt.scatter(savings.pop15, savings.sr)
plt.xlabel('Population under 15')
plt.ylabel('Savings rate')
plt.axvline(35,linestyle='dashed')
plt.plot([20,35],[lmod1.params[0]+lmod1.params[1]*20,
    lmod1.params[0]+lmod1.params[1]*35],'k-')
plt.plot([35,48],[lmod2.params[0]+lmod2.params[1]*35,
    lmod2.params[0]+lmod2.params[1]*48],'k-')


#	

def lhs (x,c): return(np.where(x < c, c-x, 0))
def rhs (x,c): return(np.where(x < c, 0, x-c))
lmod = smf.ols('sr ~ lhs(pop15,35) + rhs(pop15,35)', 
    savings).fit()
x = np.arange(20,49)
py = lmod.params[0] + lmod.params[1]*lhs(x,35) + \
    lmod.params[2]*rhs(x,35)
plt.plot(x,py,linestyle='dotted')


# ## Polynomials
#	

ethanol = sm.datasets.get_rdataset("ethanol", "lattice").data


#	

lmod2 = smf.ols('NOx ~ E + I(E**2) + C', ethanol).fit()
lmod2.sumary()


#	

-lmod2.params[1]/(2*lmod2.params[2])


#	

f2 = np.poly1d(lmod2.params[[2,1,0]])
from scipy.optimize import minimize_scalar
result = minimize_scalar(-f2)
result.x


#	

plt.scatter(ethanol.E, lmod2.resid)


#	

lmod4 = smf.ols('NOx ~ E + I(E**2) +I(E**3) + I(E**4) + C',
    ethanol).fit()
lmod4.sumary()


#	

lmodl = smf.ols('np.log(NOx) ~ E + I(E**2) + C', ethanol).fit()
lmodl.sumary()


#	

f4 = np.poly1d(lmod4.params[[4,3,2,1,0]])
fl = np.poly1d(lmodl.params[[2,1,0]])


#	

plt.scatter(ethanol.E,ethanol.NOx,label=None)
z = np.linspace(0.4,1.4)
meanC = np.mean(ethanol.C)
plt.plot(z,f2(z)+meanC*lmod2.params['C'],'k-',label='quadratic')
plt.plot(z,f4(z)+meanC*lmod4.params['C'],'k--',label='quartic')
plt.plot(z,np.exp(fl(z)+meanC*lmodl.params['C']),'k:',
    label='log-quadratic')
plt.legend()


#	

minimize_scalar(-f4).x


#	

minimize_scalar(-f4,bounds=(0.6,1.2),method="bounded").x


#	

minimize_scalar(-fl).x


#	

cmax = max(ethanol.E)
cmin = min(ethanol.E)
a=2/(cmax-cmin)
b=-(cmax+cmin)/(cmax-cmin)
sC = ethanol.E * a + b
X = np.polynomial.legendre.legvander(sC,4)
Ccol= np.array(ethanol['C']).reshape(-1,1)
X = np.concatenate((X,Ccol),axis=1)
lmodleg = sm.OLS(ethanol.NOx,X).fit()
lmodleg.sumary()


#	

np.corrcoef(X[:,1:5].T).round(3)


#	

ethanol['Ec'] = ethanol.E - 0.9
lmodc = smf.ols('NOx ~ Ec + I(Ec**2) + C', ethanol).fit()
lmodc.sumary()


#	

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X = ethanol.loc[:,['C','E']]
Xp = poly.fit_transform(X)
Xp = pd.DataFrame(Xp,columns=['Intercept','C','E','C2','E2','CE'])
lmod = sm.OLS(ethanol.NOx,Xp).fit()
lmod.sumary()


#	

ngrid = 11
Cx = np.linspace(min(ethanol.C),max(ethanol.C),ngrid)
Ey = np.linspace(min(ethanol.E),max(ethanol.E),ngrid)
X, Y = np.meshgrid(Cx,Ey)
XYpairs = np.dstack([X, Y]).reshape(-1, 2)
Xp = poly.fit_transform(XYpairs)
pv = np.dot(Xp,lmod.params)
Z = np.reshape(pv,(ngrid,ngrid))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X,Y,Z)
ax.set_xlabel("C")
ax.set_ylabel("E")
ax.set_zlabel("NOx")


# ## Splines
#	

def funeg (x): return(np.sin(2*np.pi*x**3)**3)
x = np.linspace(0., 1., 101)
y = funeg(x) + sp.stats.norm.rvs(0,0.1,101)
plt.scatter(x,y)
plt.plot(x, funeg(x))


#	

p = np.poly1d(np.polyfit(x,y,4))
plt.scatter(x,y)
plt.plot(x,p(x),'k--')
p = np.poly1d(np.polyfit(x,y,12))
plt.plot(x,p(x),'k-')


#	

from patsy import bs 
kts = [0,0.2,0.4,0.5,0.6,0.7,0.8,0.85,0.9,1]
z = sm.OLS(y,bs(x,knots=kts,include_intercept = True)).fit()
plt.scatter(x,y)
plt.plot(x, z.fittedvalues)


# ## Additive Models
#	

from statsmodels.gam.api import GLMGam, BSplines
xmat = ethanol[['C', 'E']]
bs = BSplines(xmat, df=[4, 4], degree=[3, 3])
gamod = GLMGam.from_formula('NOx ~ C + E', ethanol, 
    smoother=bs).fit()


#	

fig=gamod.plot_partial(0, cpr=True)


#	

fig=gamod.plot_partial(1, cpr=True)


# ## More Complex Models
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

    