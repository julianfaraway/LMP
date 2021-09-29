# # Models with Several Factors
# ## Two Factors with No Replication
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy import stats
import faraway.utils


#	

import faraway.datasets.composite
composite = faraway.datasets.composite.load()
composite


#	

sns.catplot(x="laser", y="strength", hue="tape",
    data=composite, kind="point")


#	

composite['Nlaser'] = np.tile([40,50,60],3)
faclevels = np.unique(composite.tape)
lineseq = ['-','--',":"]
for i in np.arange(len(faclevels)):
    j = (composite.tape == faclevels[i])
    plt.plot(composite.Nlaser[j], composite.strength[j],
             lineseq[i],label=faclevels[i])
plt.legend()
plt.xlabel("Laser")
plt.ylabel("Strength")
plt.xticks([40,50,60],["40W","50W","60W"])


#	

sns.catplot(x="tape", y="strength", hue="laser", 
    data=composite, kind="point")


#	

composite['Ntape'] = np.repeat([6.42,13,27], 3)
faclevels = np.unique(composite.laser)
for i in np.arange(len(faclevels)):
    j = (composite.laser == faclevels[i])
    plt.plot(composite.Ntape[j], composite.strength[j],
             lineseq[i],label=faclevels[i])
plt.legend()
plt.xlabel("Tape")
plt.ylabel("Strength")
plt.xticks([6.42,13,27],["slow","medium","fast"])


#	

lmod = smf.ols("strength ~ tape * laser", composite).fit()
lmod.sumary()


#	

lmod = smf.ols("strength ~ tape + laser", composite).fit()
lmod.params.index


#	

tapecoefs = np.repeat([lmod.params[2],lmod.params[1],0], 3)


#	

lasercoefs = np.tile(np.append(0, lmod.params[3:5]),3)


#	

composite['crossp'] = tapecoefs * lasercoefs
tmod = smf.ols("strength ~ tape + laser + crossp", 
    composite).fit()
tmod.sumary()


#	

sm.stats.anova_lm(tmod)


#	

cat_type = pd.api.types.CategoricalDtype(
    categories=['slow','medium','fast'],ordered=True)
composite['tape'] = composite.tape.astype(cat_type)
from patsy.contrasts import Poly
lmod = smf.ols("strength ~ C(tape,Poly) + C(laser,Poly)", 
    composite).fit()
lmod.sumary()


#	

from patsy import dmatrix
dm = dmatrix('~ C(tape,Poly) + C(laser,Poly)', composite)
np.asarray(dm).round(2)


#	

lmodn = smf.ols(
   "strength ~ np.log(Ntape) + I(np.log(Ntape)**2) + Nlaser", 
   composite).fit()
lmodn.sumary()


# ## Two Factors with Replication
#	

import faraway.datasets.pvc
pvc = faraway.datasets.pvc.load()


#	

sns.catplot(x='resin',y='psize',hue='operator', data=pvc, 
    kind="point",ci=None)


#	

pvcm = pvc.groupby(
    ['operator','resin'])[['psize']].mean().reset_index()


#	

faclevels = np.unique(pvcm.operator)
lineseq = ['-','--',":"]
for i in np.arange(len(faclevels)):
    j = (pvcm.operator == faclevels[i])
    plt.plot(pvcm.resin[j], pvcm.psize[j],
             lineseq[i],label=faclevels[i])
plt.legend(title="Operator")
plt.xlabel("resin")
plt.ylabel("Particle Size")


#	

sns.catplot(x='operator',y='psize',hue='resin', data=pvc, 
    ci=None,scale=0.5,kind="point")


#	

faclevels = np.unique(pvcm.resin)
for i in np.arange(len(faclevels)):
    j = (pvcm.resin == faclevels[i])
    plt.plot(pvcm.operator[j], pvcm.psize[j])
    plt.annotate(str(i+1), [0.95,np.ravel(pvcm.psize[j])[0]])
plt.xlabel("Operator")
plt.xticks([1,2,3])
plt.ylabel("Particle Size")


#	

pvc['operator'] = pvc['operator'].astype('category')
pvc['resin'] = pvc['resin'].astype('category')
lmod = smf.ols('psize ~ operator*resin', pvc).fit()
sm.stats.anova_lm(lmod).round(3)


#	

lmod2 = smf.ols('psize ~ operator+resin', pvc).fit()
sm.stats.anova_lm(lmod2).round(3)


#	

sm.qqplot(lmod.resid, line="q")


#	

sns.residplot(x=lmod.fittedvalues,y=lmod.resid)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")


#	

sns.swarmplot(x=pvc['operator'],y=lmod.resid)
plt.axhline(0)
plt.xlabel("Operator")
plt.ylabel("Residuals")


#	

lmod = smf.ols('psize ~ operator*resin', pvc).fit()
ii = np.arange(0,48,2)
pvce = pvc.iloc[ii,:].copy()
pvce['res'] = np.sqrt(abs(lmod.resid.loc[ii+1]))
vmod = smf.ols('res ~ operator+resin', pvce).fit()
sm.stats.anova_lm(vmod).round(3)


#	

lmod = smf.ols('psize ~ operator+resin', pvc).fit()
lmod.sumary()


#	

from statsmodels.sandbox.stats.multicomp import get_tukeyQcrit
get_tukeyQcrit(3,38) * lmod.bse[1] / np.sqrt(2)


#	

p = np.append(0,lmod.params[1:3])
np.add.outer(p,-p)


# ## Two Factors with an Interaction
#	

import faraway.datasets.warpbreaks
warpbreaks = faraway.datasets.warpbreaks.load()
warpbreaks.head()


#	

sns.boxplot(x="wool", y="breaks", data=warpbreaks)


#	

ax = sns.catplot(x='wool',y='breaks',hue='tension', 
    data=warpbreaks, ci=None,kind="point")
ax = sns.swarmplot(x='wool',y='breaks',hue='tension', 
    data=warpbreaks)
ax.legend_.remove()


#	

warpbreaks['nwool'] = np.where(warpbreaks['wool'] == 'A',0,1)
faclevels = np.unique(warpbreaks.tension)
markseq = [".","x","+"]
for i in np.arange(len(faclevels)):
    j = (warpbreaks.tension == faclevels[i])
    plt.scatter(warpbreaks.nwool[j]+0.05*i, warpbreaks.breaks[j],
                marker=markseq[i],label=faclevels[i])
plt.legend(title="Tension")
plt.xlabel("Wool")
plt.xticks([0,1],["A","B"])
plt.ylabel("Warpbreaks")


#	

wm = warpbreaks.groupby(
    ['wool','tension'])[['breaks']].mean().reset_index()
faclevels = np.unique(wm.tension)
lineseq = ['-','--',":"]
for i in np.arange(len(faclevels)):
    j = (wm.tension == faclevels[i])
    plt.plot(wm.wool[j], wm.breaks[j],lineseq[i],
             label=faclevels[i])
plt.legend(title="Tension")
plt.xlabel("Wool")
plt.ylabel("Warpbreaks")


#	

warpbreaks['ntension'] = \
    np.where(warpbreaks['tension'] == 'L',
    0,np.where(warpbreaks['tension']=='M',1,2))
faclevels = np.unique(warpbreaks.wool)
markseq = ["x","+"]
for i in np.arange(len(faclevels)):
    j = (warpbreaks.wool == faclevels[i])
    plt.scatter(warpbreaks.ntension[j]+0.1*i, 
                warpbreaks.breaks[j],
                marker=markseq[i],
                label=faclevels[i])
plt.legend(title="Wool")
plt.xlabel("Tension")
plt.xticks([0,1,2],["L","M","H"])
plt.ylabel("Warpbreaks")


#	

lineseq = ["-",":"]
for i in np.arange(len(faclevels)):
    j = (wm.wool == faclevels[i])
    plt.plot(wm.tension[j], wm.breaks[j],
             lineseq[i],label=faclevels[i])
plt.legend(title="Wool")
plt.xlabel("Tension")
plt.ylabel("Warpbreaks")


#	

lmod = smf.ols('breaks ~ wool * tension', warpbreaks).fit()
fig, ax = plt.subplots()
sns.regplot(x=lmod.fittedvalues,y=lmod.resid,
    scatter_kws={'alpha':0.3}, ci=None, ax=ax)
ax.set_xlim(np.array(ax.get_xlim())+[-1,1])
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")


#	

lmod = smf.ols('np.sqrt(breaks) ~ wool * tension', 
    warpbreaks).fit()
fig, ax = plt.subplots()
sns.regplot(x=lmod.fittedvalues,y=lmod.resid,
    scatter_kws={'alpha':0.3}, ci=None, ax=ax)
ax.set_xlim(np.array(ax.get_xlim())+[-0.1,0.1])
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")


#	

sm.stats.anova_lm(lmod).round(4)


#	

lmod.sumary()


#	

lmod = smf.ols(
    'np.sqrt(breaks) ~ wool:tension-1', warpbreaks).fit()
lmod.sumary()


#	

get_tukeyQcrit(6,48) * lmod.bse[0]


#	

import itertools
dp = set(itertools.combinations(range(0,6),2))
dcoef = []
namdiff = []
for cp in dp:
    dcoef.append(lmod.params[cp[0]] - lmod.params[cp[1]])
    namdiff.append(lmod.params.index[cp[0]] + '-' + \
                   lmod.params.index[cp[1]])
thsd = pd.DataFrame({'Difference':dcoef},index=namdiff)
thsd["lb"] = thsd.Difference - get_tukeyQcrit(6,48) * lmod.bse[0]
thsd["ub"] = thsd.Difference + get_tukeyQcrit(6,48) * lmod.bse[0]
thsd.round(3)


# ## Larger Factorial Experiments
#	

import faraway.datasets.speedo
speedo = faraway.datasets.speedo.load()
speedo


#	

lmod = smf.ols('y ~ h+d+l+b+j+f+n+a+i+e+m+c+k+g+o', speedo).fit()
lmod.sumary()


#	

import patsy
dm = patsy.dmatrix('~ h+d+l+b+j+f+n+a+i+e+m+c+k+g+o', speedo)
np.asarray(dm)


#	

ii = np.argsort(lmod.params[1:])
scoef = np.array(lmod.params[1:])[ii]
lcoef = (speedo.columns[:-1])[ii]
n = len(scoef)
qq = stats.norm.ppf(np.arange(1,n+1)/(n+1))
fig, ax = plt.subplots()
ax.scatter(qq, scoef,s=1)
for i in range(len(qq)):
    ax.annotate(lcoef[i], (qq[i],scoef[i]))
plt.xlabel("Normal Quantiles")
plt.ylabel("Sorted coefficients")


#	

n = len(scoef)
qq = stats.norm.ppf((n + np.arange(1,n+1))/(2*n+1))
acoef = np.abs(lmod.params[1:])
ii = np.argsort(acoef)
acoef = acoef[ii]
lcoef = (speedo.columns[:-1])[ii]
fig, ax = plt.subplots()
ax.scatter(qq, acoef,s=1)
for i in range(len(qq)):
    ax.annotate(lcoef[i], (qq[i],acoef[i]))
plt.xlabel("Normal Half Quantiles")
plt.ylabel("Sorted absolute coefficients")


# ## Exercises
#	
#	


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

    
