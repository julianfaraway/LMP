# # Diagnostics
# ## Checking Error Assumptions
# ### Constant Variance
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils
import seaborn as sns


#	

n = 50
np.random.seed (123)
x = np.random.sample(n)


#	

y = np.random.normal(size=n)
plt.scatter(x,y)
plt.title("No problem plot")
plt.axhline(0)


#	

plt.scatter(x,y*x)
plt.title("Increasing variance")
plt.axhline(0)


#	

y = np.cos(2*x*np.pi) + np.random.normal(size=n)
plt.scatter(x,y)
plt.title("Lack of fit plot")
sx = np.sort(x)
plt.plot(sx,np.cos(2*sx*np.pi),'k-')
plt.axhline(0)


#	

import faraway.datasets.savings
savings = faraway.datasets.savings.load()
lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings).fit()


#	

plt.scatter(lmod.fittedvalues, lmod.resid)
plt.ylabel("Residuals")
plt.xlabel("Fitted values")
plt.axhline(0)


#	

plt.scatter(lmod.fittedvalues, np.sqrt(abs(lmod.resid)))
plt.ylabel("sqrt |Residuals|")
plt.xlabel("Fitted values")


#	

ddf = pd.DataFrame({'x':lmod.fittedvalues,
   'y':np.sqrt(abs(lmod.resid))})
dmod = smf.ols('y ~ x',ddf).fit()
dmod.sumary()


#	

plt.scatter(savings.pop15, lmod.resid)
plt.xlabel("%pop under 15")
plt.ylabel("Residuals")
plt.axhline(0)


#	

plt.scatter(savings.pop75, lmod.resid)
plt.xlabel("%pop over 75")
plt.ylabel("Residuals")
plt.axhline(0)


#	

numres = lmod.resid[savings.pop15 > 35]
denres = lmod.resid[savings.pop15 < 35]
fstat = np.var(numres,ddof=1)/np.var(denres,ddof=1)
2*(1-sp.stats.f.cdf(fstat,len(numres)-1,len(denres)-1))


#	

import faraway.datasets.galapagos
galapagos = faraway.datasets.galapagos.load()
gmod = smf.ols(
    'Species ~ Area + Elevation + Nearest + Scruz + Adjacent', 
    galapagos).fit()
plt.scatter(gmod.fittedvalues, gmod.resid)
plt.ylabel("Residuals")
plt.xlabel("Fitted values")
plt.axhline(0)


#	

gmod = smf.ols(
   'np.sqrt(Species) ~ Area + Elevation + Nearest + Scruz + Adjacent', 
   galapagos).fit()
plt.scatter(gmod.fittedvalues, gmod.resid)
plt.ylabel("Residuals")
plt.xlabel("Fitted values")
plt.axhline(0)


# ### Normality
#	

sm.qqplot(lmod.resid, line="q")


#	

plt.hist(lmod.resid)
plt.xlabel("Residuals")


#	

fig, axs = plt.subplots(2, 2, sharex=True)
sm.qqplot(np.random.normal(size=50),line="q",ax = axs[0,0])
sm.qqplot(np.exp(np.random.normal(size=50)),line="q",ax=axs[0,1])
sm.qqplot(np.random.standard_t(1,size=50),line="q",ax=axs[1,0])
sm.qqplot(np.random.sample(size=50),line="q",ax=axs[1,1])
plt.tight_layout()


#	

sp.stats.shapiro(lmod.resid)


# ### Correlated Errors
#	

import faraway.datasets.globwarm
globwarm = faraway.datasets.globwarm.load()
lmod=smf.ols('nhtemp ~ wusa + jasper + westgreen + chesapeake + \
    tornetrask + urals + mongolia + tasman', globwarm).fit()


#	

plt.scatter(globwarm.year[lmod.resid.keys()], lmod.resid)
plt.axhline(0)


#	

plt.scatter(lmod.resid.iloc[:-1],lmod.resid.iloc[1:])
plt.axhline(0,alpha=0.5)
plt.axvline(0,alpha=0.5)


#	

sm.stats.stattools.durbin_watson(lmod.resid)


# ## Finding Unusual Observations
# ### Leverage
#	

lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings).fit()
diagv = lmod.get_influence()
hatv = pd.Series(diagv.hat_matrix_diag, savings.index)
hatv.sort_values().tail()


#	

np.sum(hatv)


#	

n=50
ix = np.arange(1,n+1)
halfq = sp.stats.norm.ppf((n+ix)/(2*n+1)),
plt.scatter(halfq, np.sort(hatv))
plt.annotate("Libya",(2.1,0.53))
plt.annotate("USA", (1.9,0.33))


#	

rstandard = diagv.resid_studentized_internal
sm.qqplot(rstandard)


# ### Outliers
#	

np.random.seed(123)
testdata = pd.DataFrame({'x' : np.arange(1,11), 
    'y' : np.arange(1,11) + np.random.normal(size=10)})
p1 = pd.DataFrame({'x': [5.5], 'y':[12]})
alldata = testdata.append(p1,ignore_index=True)


#	

marksize = np.ones(11)
marksize[10] = 3
plt.scatter(alldata.x, alldata.y, s= marksize*5)
slope, intercept = np.polyfit(testdata.x, testdata.y,1)
plt.plot(testdata.x, intercept + slope * testdata.x)
slope, intercept = np.polyfit(alldata.x, alldata.y,1)
plt.plot(alldata.x, intercept + slope * alldata.x, '--')


#	

p1 = pd.DataFrame({'x': [15], 'y':[15.1]})
alldata = testdata.append(p1,ignore_index=True)
plt.scatter(alldata.x, alldata.y, s= marksize*5)
slope, intercept = np.polyfit(testdata.x, testdata.y,1)
plt.plot(testdata.x, intercept + slope * testdata.x)
slope, intercept = np.polyfit(alldata.x, alldata.y,1)
plt.plot(alldata.x, intercept + slope * alldata.x, '--')


#	

p1 = pd.DataFrame({'x': [15], 'y':[5.1]})
alldata = testdata.append(p1,ignore_index=True)
plt.scatter(alldata.x, alldata.y, s= marksize*5)
slope, intercept = np.polyfit(testdata.x, testdata.y,1)
plt.plot(testdata.x, intercept + slope * testdata.x)
slope, intercept = np.polyfit(alldata.x, alldata.y,1)
plt.plot(alldata.x, intercept + slope * alldata.x, '--')


#	

stud = pd.Series(diagv.resid_studentized_external, savings.index)
(pd.Series.idxmax(abs(stud)), np.max(abs(stud)))


#	

abs(sp.stats.t.ppf(0.05/(2*50),44))


#	

import faraway.datasets.star
star = faraway.datasets.star.load()
plt.scatter(star.temp, star.light)


#	

lmod = smf.ols('light ~ temp',star).fit()
xr = np.array([np.min(star.temp),np.max(star.temp)])
plt.plot(xr, lmod.params[0] + lmod.params[1]*xr)
plt.xlabel("log(Temperature)")
plt.ylabel("log(Light Intensity)")


#	

stud = lmod.get_influence().resid_studentized_external
np.min(stud), np.max(stud)


#	

lmodr = smf.ols('light ~ temp',star[star.temp > 3.6]).fit()
plt.plot(xr, lmodr.params[0] + lmodr.params[1]*xr,'k--')


# ### Influential Observations
#	

lmod = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', savings).fit()
diagv = lmod.get_influence()
cooks = pd.Series(diagv.cooks_distance[0], savings.index)
n=50
ix = np.arange(1,n+1)
halfq = sp.stats.norm.ppf((n+ix)/(2*n+1)),
plt.scatter(halfq, np.sort(cooks))


#	

cooks.sort_values().iloc[-5:]


#	

lmodi = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', 
    savings[cooks < 0.2]).fit()
pd.DataFrame({'with':lmod.params,'without':lmodi.params})


#	

p15d = diagv.dfbetas[:,1]
plt.scatter(np.arange(1,51),p15d)
plt.axhline(0)
ix = 22
plt.annotate(savings.index[ix],(ix, p15d[ix]))
ix = 48
plt.annotate(savings.index[ix],(ix, p15d[ix]))


#	

lmodj = smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', 
    savings.drop(['Japan'])).fit()
lmodj.sumary()


# ## Checking the Structure of the Model
#	

d = smf.ols('sr ~ pop75 + dpi + ddpi', savings).fit().resid
m = smf.ols('pop15 ~ pop75 + dpi + ddpi', savings).fit().resid
plt.scatter(m,d)
plt.xlabel("pop15 residuals")
plt.ylabel("sr residuals")
plt.plot([-10,8], [-10*lmod.params[1], 8*lmod.params[1]])


#	

np.polyfit(m,d,deg=1)[0], lmod.params[1]


#	

pr = lmod.resid + savings.pop15*lmod.params[1]
plt.scatter(savings.pop15, pr)
plt.xlabel("pop15")
plt.ylabel("partial residuals")
plt.plot([20,50], [20*lmod.params[1], 50*lmod.params[1]])


#	

smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', 
    savings[savings.pop15 > 35]).fit().sumary()


#	

smf.ols('sr ~ pop15 + pop75 + dpi + ddpi', 
    savings[savings.pop15 < 35]).fit().summary()


#	

savings['age'] = np.where(savings.pop15 > 35, 'young', 'old')
sns.lmplot('ddpi','sr',data=savings, hue='age',legend_out=False)


#	

sns.lmplot('ddpi','sr',data=savings, col='age')


# ## Discussion
# ## Excercises

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

    