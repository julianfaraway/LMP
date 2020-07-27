# # Insurance Redlining --- A Complete Example
# ## Ecological Correlation
#	

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import faraway.utils


#	

import faraway.datasets.eco
eco = faraway.datasets.eco.load()
plt.scatter(eco.usborn, eco.income)
plt.ylabel("Income")
plt.xlabel("Fraction US born")


#	

lmod = sm.OLS(eco.income, sm.add_constant(eco.usborn)).fit()
lmod.sumary()


#	

plt.scatter(eco.usborn, eco.income)
plt.ylabel("Income")
plt.xlabel("Fraction US born")
xr = np.array([0,1])
plt.plot(xr,lmod.params[0] + xr*lmod.params[1])


# ## Initial Data Analysis
#	

import faraway.datasets.chredlin
chredlin = faraway.datasets.chredlin.load()
chredlin.head()


#	

chredlin.drop('zip',1).describe().round(2)


#	

chredlin.side.value_counts()


#	

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = \
    plt.subplots(nrows=2, ncols=3, sharey=True)
ax1.scatter(chredlin.race, chredlin.involact)
ax1.set(title="Race")
ax1.set_ylabel("Involact")

ax2.scatter(chredlin.fire, chredlin.involact)
ax2.set(title="Fire")

ax3.scatter(chredlin.theft, chredlin.involact)
ax3.set(title="Theft")

ax4.scatter(chredlin.age, chredlin.involact)
ax4.set(title="Age")
ax4.set_ylabel("Involact")

ax5.scatter(chredlin.income, chredlin.involact)
ax5.set(title="Income")

ax6.scatter(chredlin.side, chredlin.involact)
ax6.set(title="Side")

fig.tight_layout()


#	

lmod = sm.OLS(chredlin.involact, 
    sm.add_constant(chredlin.race)).fit()
lmod.sumary()


#	

fig = plt.figure(figsize=(6,3))

ax1 = fig.add_subplot(121)

ax1.scatter(chredlin.race, chredlin.fire)
ax1.set_xlabel("Race")
ax1.set_ylabel("Fire")

ax2 = fig.add_subplot(122)
ax2.scatter(chredlin.race, chredlin.theft)
ax2.set_xlabel("Race")
ax2.set_ylabel("Theft")

fig.tight_layout()


# ## Full Model and Diagnostics
#	

lmod = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)',
     chredlin).fit()
lmod.sumary()


#	

plt.scatter(lmod.fittedvalues, lmod.resid)
plt.ylabel("Residuals")
plt.xlabel("Fitted values")
plt.axhline(0)


#	

sm.qqplot(lmod.resid, line="q")


#	

pr = lmod.resid + chredlin.race*lmod.params['race']
plt.scatter(chredlin.race, pr)
plt.xlabel("Race")
plt.ylabel("partial residuals")
xr = np.array(plt.xlim())
plt.plot(xr, xr*lmod.params['race'])


#	

pr = lmod.resid + chredlin.fire*lmod.params['fire']
plt.scatter(chredlin.fire, pr)
plt.xlabel("Fire")
plt.ylabel("partial residuals")
xr = np.array(plt.xlim())
plt.plot(xr, xr*lmod.params['fire'])


# ## Sensitivity Analysis
#	

import itertools
inds = [1, 2, 3, 4]
clist = []
for i in range(0, len(inds)+1):
    clist.extend(itertools.combinations(inds, i))


#	

X = chredlin.iloc[:,[0,1,2,3,5]].copy()
X.loc[:,'income'] = np.log(chredlin['income'])
betarace = []
pvals = []
for k in range(0, len(clist)):
    lmod = sm.OLS(chredlin.involact, 
       sm.add_constant(X.iloc[:,np.append(0,clist[k])])).fit()
    betarace.append(lmod.params[1])
    pvals.append(lmod.pvalues[1])


#	

vlist = ['race']
varnames = np.array(['race','fire','theft','age','logincome'])
for k in range(1, len(clist)):
    vlist.append('+'.join(varnames[np.append(0,clist[k])]))


#	

pd.DataFrame({'beta':np.round(betarace,4), 
    'pvals':np.round(pvals,4)}, index=vlist)


#	

lmod = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)', 
    chredlin).fit()
diagv = lmod.get_influence()
min(diagv.dfbetas[:,1])


#	

plt.scatter(diagv.dfbetas[:,2], diagv.dfbetas[:,3])
plt.xlabel("Change in Fire")
plt.ylabel("Change in Theft")
plt.axhline(0)
plt.axvline(0)


#	

sm.graphics.influence_plot(lmod)


#	

chredlin.iloc[[5, 23],:]


#	

ch45 = chredlin.drop(chredlin.index[[5,23]])
lmod45 = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)',
     ch45).fit()
lmod45.sumary()


#	

lmodalt = smf.ols('involact ~ race + fire + np.log(income)',
    ch45).fit()
lmodalt.sumary()


# ## Discussion
#	

lmods = smf.ols('involact ~ race + fire + theft + age', 
    chredlin.loc[chredlin.side == 's',:]).fit()
lmods.sumary()


#	

lmodn = smf.ols('involact ~ race + fire + theft + age', 
    chredlin.loc[chredlin.side == 'n',:]).fit()
lmodn.sumary()


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

    