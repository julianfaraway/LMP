# # Categorical Predictors
# ## A Two-Level Factor
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

import faraway.datasets.sexab
sexab = faraway.datasets.sexab.load()
sexab.head()


#	

lfuncs = ['min','median','max']
sexab.groupby('csa').agg({'cpa': lfuncs,'ptsd': lfuncs}).round(1)


#	

sns.boxplot(x="csa", y="ptsd", data=sexab)


#	

sns.pairplot(x_vars="cpa", y_vars="ptsd", data=sexab, 
    hue="csa", markers=["s","o"])


#	

stats.ttest_ind(sexab.ptsd[sexab.csa == 'Abused'], 
    sexab.ptsd[sexab.csa == 'NotAbused'])


#	

df1 = (sexab.csa == 'Abused').astype(int)
df2 = (sexab.csa == 'NotAbused').astype(int)
X = np.column_stack((df1,df2))
lmod = sm.OLS(sexab.ptsd,sm.add_constant(X)).fit()
lmod.sumary()


#	

sm.add_constant(X)[[0,44,45,75],:]


#	

lmod = sm.OLS(sexab.ptsd,sm.add_constant(df2)).fit()
lmod.sumary()


#	

lmod = sm.OLS(sexab.ptsd,X).fit()
lmod.sumary()


#	

lmod = smf.ols('ptsd ~ csa', sexab).fit()
lmod.sumary()


#	

sexab.csa.dtype, sexab.ptsd.dtype


#	

sac = pd.concat([sexab,pd.get_dummies(sexab.csa)],axis=1)
sac.iloc[[0,44,45,75],:]


#	

lmod = smf.ols('ptsd ~ Abused', sac).fit()
lmod.sumary()


#	

lmod = smf.ols(
    'ptsd ~ C(csa,Treatment(reference="NotAbused"))',
    sexab).fit()
lmod.sumary()



# ## Factors and Quantitative Predictors
#	

lmod4 = smf.ols('ptsd ~ csa*cpa', sexab).fit()
lmod4.sumary()


#	

import patsy
patsy.dmatrix('~ csa*cpa', sexab)[[0,44,45,75],]


#	

abused = (sexab.csa == "Abused")
plt.scatter(sexab.cpa[abused], sexab.ptsd[abused], marker='x')
xl,xu = [-3, 9]
a, b = (lmod4.params[0], lmod4.params[2])
plt.plot([xl,xu], [a+xl*b,a+xu*b])
plt.scatter(sexab.cpa[~abused], sexab.ptsd[~abused], marker='o')
a, b = (lmod4.params[0]+lmod4.params[1], 
        lmod4.params[2]+lmod4.params[3])
plt.plot([xl,xu], [a+xl*b,a+xu*b])


#	

lmod3 = smf.ols('ptsd ~ csa+cpa', sexab).fit()
lmod3.sumary()


#	

plt.scatter(sexab.cpa[abused], sexab.ptsd[abused], marker='x')
xl,xu = [-3, 9]
a, b = (lmod3.params[0], lmod3.params[2])
plt.plot([xl,xu], [a+xl*b,a+xu*b])
plt.scatter(sexab.cpa[~abused], sexab.ptsd[~abused], marker='o')
a, b = (lmod3.params[0]+lmod4.params[1], lmod3.params[2])
plt.plot([xl,xu], [a+xl*b,a+xu*b])


#	

lmod3.conf_int().round(2)


#	

plt.scatter(lmod3.fittedvalues[abused], lmod3.resid[abused], 
    marker='x')
plt.scatter(lmod3.fittedvalues[~abused], lmod3.resid[~abused], 
    marker='o')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.axhline(0)


#	

lmod1 = smf.ols('ptsd ~ cpa', sexab).fit()
lmod1.sumary()


# ## Interpretation with Interaction Terms
#	

import faraway.datasets.whiteside
whiteside = faraway.datasets.whiteside.load()


#	

sns.lmplot(x="Temp", y="Gas", col="Insul", data=whiteside)


#	

whiteside['Insul'] = pd.Categorical(whiteside['Insul'],
                         categories=['Before','After'])
whiteside['Insul'].dtype


#	

lmod = smf.ols('Gas ~ Temp*Insul', whiteside).fit()
lmod.sumary()


#	

whiteside.Temp.mean()


#	

whiteside['cTemp'] = whiteside.Temp - whiteside.Temp.mean()
lmod = smf.ols('Gas ~ cTemp*Insul', whiteside).fit()
lmod.sumary()


#	

lmod = smf.ols('np.log(Gas) ~ Temp*Insul', whiteside).fit()
lmod.sumary()


#	

lmod = smf.ols('np.log(Gas) ~ Temp+Insul', whiteside).fit()
lmod.sumary()


# ## Factors with More Than Two Levels
#	

import faraway.datasets.fruitfly
ff = faraway.datasets.fruitfly.load()


#	

sns.pairplot(x_vars="thorax", y_vars="longevity", data=ff, 
    hue="activity",height=5,markers=["o",".","P","X","v"])


#	

sns.lmplot(x="thorax", y="longevity", data=ff, 
    col="activity",height=5,col_wrap=3)


#	

lmod = smf.ols('longevity ~ thorax*activity', ff).fit()
lmod.sumary()


#	

mm = patsy.dmatrix('~ thorax*activity', ff)
ii = (1, 25, 49, 75, 99)
p = pd.DataFrame(mm[ii,:],index=ii,columns=lmod.params.index)
p.iloc[:,[0,1,5,6]]


#	

sm.stats.anova_lm(lmod)


#	

lmod = smf.ols('longevity ~ thorax+activity', ff).fit()
lmod.sumary()


#	

g = sns.residplot(x=lmod.fittedvalues, y=lmod.resid)
g.set(xlabel="Fitted values",ylabel="Residuals")


#	

lmod = smf.ols('np.log(longevity) ~ thorax+activity', ff).fit()
g = sns.residplot(x=lmod.fittedvalues, y=lmod.resid)
g.set(xlabel="Fitted values",ylabel="Residuals")


#	

np.round(np.exp(lmod.params[1:5]),2)


#	

lmod = smf.ols('thorax ~ activity', ff).fit()
sm.stats.anova_lm(lmod)


#	

lmod = smf.ols('np.log(longevity) ~ activity', ff).fit()
lmod.sumary()


# ## Alternative Codings of Qualitative Predictors
#	

smod = smf.ols('np.log(longevity) ~ C(activity,Sum)', ff).fit()
smod.sumary()


#	

np.sum(smod.params[1:])


#	

from patsy.contrasts import Treatment
levels = [1,2,3,4]
contrast = Treatment(reference=0).code_without_intercept(levels)
print(contrast.matrix)


#	

from patsy.contrasts import Sum
contrast = Sum().code_without_intercept(levels)
print(contrast.matrix)


#	

mm = patsy.dmatrix('~ C(activity,Sum)', ff)
ii = [1, 25, 49, 75, 99]
pd.DataFrame(mm[ii,:], index=ff.activity.iloc[ii], 
    columns=['intercept','isolated','low','high','one'])


# ## Exercises
#	

    tgdf = sm.datasets.get_rdataset("ToothGrowth")
    tg = tgdf.data


#	

    print(tgdf.__doc__)



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

    