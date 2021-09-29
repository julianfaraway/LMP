# # Experiments with Blocks
# ## Randomized Block Design
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

import faraway.datasets.oatvar
oatvar = faraway.datasets.oatvar.load()
oatvar.pivot(index = 'variety', columns='block', values='yield')


#	

sns.boxplot(x="variety", y="yield", data=oatvar)
sns.boxplot(x="block", y="yield", data=oatvar)
sns.catplot(x='variety',y='yield',hue='block', data=oatvar, 
   kind='point')
sns.catplot(x='block',y='yield',hue='variety', data=oatvar, 
   kind='point')


#	

faclevels = np.unique(oatvar.block)
oatvar["variety"] = oatvar["variety"].astype('category')
oatvar["block"] = oatvar["block"].astype('category')
for i in np.arange(len(faclevels)):
    j = (oatvar.block == faclevels[i])
    plt.plot(oatvar.variety.cat.codes[j], oatvar.grams[j])
    plt.annotate(faclevels[i], 
        [-0.2,np.ravel(oatvar.grams[j])[0]])
    plt.annotate(faclevels[i], 
        [7.05,np.ravel(oatvar.grams[j])[7]])
plt.xlabel("Variety")
plt.xticks(np.arange(0,8),np.arange(1,9))
plt.ylabel("Yield")


#	

faclevels = np.unique(oatvar.variety)
for i in np.arange(len(faclevels)):
    j = (oatvar.variety == faclevels[i])
    plt.plot(oatvar.block.cat.codes[j], oatvar.grams[j])
    plt.annotate(faclevels[i], 
                 [-0.15,np.ravel(oatvar.grams[j])[0]])
    plt.annotate(faclevels[i], 
                 [4.05,np.ravel(oatvar.grams[j])[4]])
plt.xlabel("Block")
plt.xticks(np.arange(0,5),oatvar.block.cat.categories)
plt.ylabel("Yield")


#	

lmod = smf.ols('grams ~ variety + block', oatvar).fit()
sm.stats.anova_lm(lmod).round(3)


#	

oat1 = oatvar.iloc[1:,]
lmod = smf.ols('grams ~ variety + block', oat1).fit()
sm.stats.anova_lm(lmod).round(4)


#	

lmod = smf.ols('grams ~ block + variety', oat1).fit()
sm.stats.anova_lm(lmod).round(4)


#	

sm.stats.anova_lm(lmod,typ=3).round(4)


#	

lmod = smf.ols('grams ~ variety + block', oatvar).fit()
plt.scatter(lmod.fittedvalues, lmod.resid)
plt.axhline(0)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")


#	

p=sm.qqplot(lmod.resid, line="q")


#	

varcoefs = np.append([0.],lmod.params[1:8])
varcoefs = np.repeat(varcoefs,5)
blockcoefs = np.append([0.],lmod.params[8:12])
blockcoefs = np.tile(blockcoefs,8)
oatvar['crossp'] = varcoefs * blockcoefs
tmod = smf.ols("grams ~ variety + block + crossp", oatvar).fit()
sm.stats.anova_lm(tmod).round(3)


#	

lmcrd = smf.ols("grams ~ variety", oatvar).fit()
lmcrd.scale/lmod.scale


# ## Latin Squares
#	

import faraway.datasets.abrasion
abrasion = faraway.datasets.abrasion.load()
abrasion.pivot(index = 'run', columns='position', values='wear')


#	

abrasion.pivot(index = 'run', columns='position', 
               values='material')


#	

sns.catplot(x='run',y='wear',hue='position', data=abrasion, 
    kind='point',scale=0.5)
sns.catplot(x='run',y='wear',hue='material', data=abrasion, 
    kind='point',scale=0.5)


#	

faclevels = np.unique(abrasion.position)
lineseq = ['-','--','-.',':']
for i in np.arange(len(faclevels)):
    j = (abrasion.position == faclevels[i])
    plt.plot(abrasion.run[j], abrasion.wear[j],
             lineseq[i],label=faclevels[i])
plt.legend(title="Position")
plt.xlabel("Run")
plt.xticks(np.arange(1,5))
plt.ylabel("Wear")


#	

faclevels = np.unique(abrasion.run)
lineseq = ['-','--','-.',':']
for i in np.arange(len(faclevels)):
    j = (abrasion.run == faclevels[i])
    plt.plot(abrasion.position[j], abrasion.wear[j],
             lineseq[i],label=faclevels[i])
plt.legend(title="Run")
plt.xlabel("Position")
plt.xticks(np.arange(1,5))
plt.ylabel("Wear")


#	

abrasion["run"] = abrasion["run"].astype('category')
abrasion["position"] = abrasion["position"].astype('category')
lmod = smf.ols('wear ~ run + position + material', abrasion).fit()
sm.stats.anova_lm(lmod,typ=3).round(3)


#	

lmod.sumary()


#	

from statsmodels.sandbox.stats.multicomp import get_tukeyQcrit
get_tukeyQcrit(4,6)*lmod.bse[1]/np.sqrt(2)


#	

treatname = 'material'
treatlevs = ['A','B','C','D']


#	

ii = lmod.params.index.str.match(treatname)
mcoefs = np.append([0],lmod.params[ii])
import itertools
dp = set(itertools.combinations(range(0,len(treatlevs)),2))
dcoef = []
namdiff = []
for cp in dp:
    dcoef.append(mcoefs[cp[0]] - mcoefs[cp[1]])
    namdiff.append(treatlevs[cp[0]] + '-' + treatlevs[cp[1]])
thsd = pd.DataFrame({'Difference':dcoef},index=namdiff)
cvband = get_tukeyQcrit(len(treatlevs), 
         lmod.df_resid) * lmod.bse[1]/np.sqrt(2)
thsd["lb"] = thsd.Difference - cvband 
thsd["ub"] = thsd.Difference + cvband 
thsd.round(2)


#	

lmodr = smf.ols('wear ~ material', abrasion).fit()
lmodr.scale/lmod.scale


# ## Balanced Incomplete Block Design
#	

import faraway.datasets.rabbit
rabbit = faraway.datasets.rabbit.load()
pt = rabbit.pivot(index = 'treat', columns='block', values='gain')
pt.replace(np.nan," ", regex=True)


#	

sns.swarmplot(x='block',y='gain',hue='treat',data=rabbit)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#	

sns.swarmplot(x='treat',y='gain',hue='block',data=rabbit)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#	

rabbit["treat"] = rabbit["treat"].astype('category')
rabbit["block"] = rabbit["block"].astype('category')
plt.scatter(rabbit.block.cat.codes,rabbit.gain,color="white")
for i in np.arange(0,len(rabbit.gain)):
    plt.annotate(rabbit.treat.iloc[i],
       (rabbit.block.cat.codes.iloc[i],rabbit.gain.iloc[i]))
plt.xticks(np.arange(0,10),rabbit.block.cat.categories)
plt.ylabel("Gain")
plt.xlabel("Block")


#	

plt.scatter(rabbit.treat.cat.codes,rabbit.gain,color="white")
for i in np.arange(0,len(rabbit.gain)):
    plt.annotate(rabbit.block.iloc[i],
        (rabbit.treat.cat.codes.iloc[i],rabbit.gain.iloc[i]))
plt.xticks(np.arange(0,6),rabbit.treat.cat.categories)
plt.ylabel("Gain")
plt.xlabel("Treatment")


#	

lmod = smf.ols('gain ~ treat + block', rabbit).fit()
sm.stats.anova_lm(lmod,typ=3).round(3)


#	

sns.residplot(x=lmod.fittedvalues, y=lmod.resid)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
p=sm.qqplot(lmod.resid, line="q")


#	

lmod.sumary()


#	

mcoefs = np.append([0],lmod.params[1:6])
nmats = [chr(i) for i in range(ord('a'),ord('f')+1)]
p = len(mcoefs)
dp = set(itertools.combinations(range(0,p),2))
dcoef = []
namdiff = []
for cp in dp:
    dcoef.append(mcoefs[cp[0]] - mcoefs[cp[1]])
    namdiff.append(nmats[cp[0]] + '-' + nmats[cp[1]])
thsd = pd.DataFrame({'Difference':dcoef},index=namdiff)
thsd["lb"] = thsd.Difference - \
    get_tukeyQcrit(p,lmod.df_resid) * lmod.bse[1]/np.sqrt(2)
thsd["ub"] = thsd.Difference + \
    get_tukeyQcrit(p,lmod.df_resid) * lmod.bse[1]/np.sqrt(2)
thsd.round(2)


#	

lmodr = smf.ols('gain ~ treat', rabbit).fit()
lmodr.scale/lmod.scale


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

    