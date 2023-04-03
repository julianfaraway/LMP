# # Introduction
# ## Before You Start
# ## Initial Data Analysis
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import statsmodels.formula.api as smf


#	

import faraway.datasets.pima
pima = faraway.datasets.pima.load()
pima.head()


#	

print(faraway.datasets.pima.DESCR)


#	

pima.describe().round(1)


#	

pima['diastolic'].sort_values().head()


#	

np.sum(pima['diastolic'] == 0)


#	

pima.replace({'diastolic' : 0, 'triceps' : 0, 'insulin' : 0, 
    'glucose' : 0, 'bmi' : 0}, np.nan, inplace=True)


#	

pima['test'] = pima['test'].astype('category')
pima['test'] = pima['test'].cat.rename_categories(
    ['Negative','Positive'])
pima['test'].value_counts()


#	

sns.displot(pima.diastolic.dropna(), kde=True)


#	

pimad = pima.diastolic.dropna().sort_values()
sns.lineplot(x=range(0, len(pimad)), y=pimad)


#	

sns.scatterplot(x='diastolic',y='diabetes',data=pima, s=20)


#	

sns.boxplot(x="test", y="diabetes", data=pima)


#	

sns.scatterplot(x="diastolic", y="diabetes", data=pima, 
    style="test", alpha=0.3)


#	

sns.relplot(x="diastolic", y="diabetes", data=pima, col="test")


# ## When to Use Linear Modeling
# ## History
#	

import faraway.datasets.manilius
manilius = faraway.datasets.manilius.load()
manilius.head()


#	

moon3 = manilius.groupby('group').sum()
moon3


#	

moon3['intercept'] = [9]*3
np.linalg.solve(moon3[['intercept','sinang','cosang']],
    moon3['arc'])


#	

mod = smf.ols('arc ~ sinang + cosang', manilius).fit()
mod.params


#	

import faraway.datasets.families
families = faraway.datasets.families.load()
sns.scatterplot(x='midparentHeight', y='childHeight',
    data=families, s=20)


#	

mod = smf.ols('childHeight ~ midparentHeight', families).fit()
mod.params


#	

cor = sp.stats.pearsonr(families['childHeight'],
    families['midparentHeight'])[0]
sdy = np.std(families['childHeight'])
sdx = np.std(families['midparentHeight'])
beta = cor*sdy/sdx
alpha = np.mean(families['childHeight']) - \
    beta*np.mean(families['midparentHeight'])
np.round([alpha,beta],2)


#	

beta1 = sdy/sdx
alpha1 = np.mean(families['childHeight']) - \
    beta1*np.mean(families['midparentHeight'])


#	

sns.lmplot(x='midparentHeight', y='childHeight', data=families, 
    ci=None, scatter_kws={'s':2})
xr = np.array([64,76])
plt.plot(xr, alpha1 + xr*beta1,'--')


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

    