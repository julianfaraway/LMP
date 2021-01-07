# # One Factor Models
# ## The Model
# ## An Example
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

import faraway.datasets.coagulation
coagulation = faraway.datasets.coagulation.load()
coagulation.head()


#	

sns.boxplot(x="diet", y="coag", data=coagulation)


#	

sns.swarmplot(x="diet", y="coag", data=coagulation)


#	

lmod = smf.ols("coag ~ diet", coagulation).fit()
lmod.sumary()


#	

import patsy
p = patsy.dmatrix('~ diet', coagulation)
p[[0,4,10,16],:]


#	

sm.stats.anova_lm(lmod)


#	

lmodi = smf.ols("coag ~ diet-1", coagulation).fit()
lmodi.sumary()


#	

lmodnull = smf.ols("coag ~ 1", coagulation).fit()
sm.stats.anova_lm(lmodnull, lmod)


#	

from patsy.contrasts import Sum
lmods = smf.ols("coag ~ C(diet,Sum)", coagulation).fit()
lmods.sumary()


# ## Diagnostics
#	

p = sns.scatterplot(lmod.fittedvalues + \
   np.random.uniform(-0.1,0.1, len(coagulation)), lmod.resid)
p.axhline(0,ls='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")


#	

sm.qqplot(lmod.resid, line="q")


#	

coagulation['meds'] = \
    coagulation.groupby('diet').transform(np.median)
coagulation['mads'] = abs(coagulation.coag - coagulation.meds)
lmodb = smf.ols('mads ~ diet', coagulation).fit()
sm.stats.anova_lm(lmodb)


#	

stats.bartlett(coagulation.coag[coagulation.diet == "A"],
               coagulation.coag[coagulation.diet == "B"],
               coagulation.coag[coagulation.diet == "C"],
               coagulation.coag[coagulation.diet == "D"])


# ## Pairwise Comparisons
#	

lmod.params[1] + \
   np.array([-1, 1]) * stats.t.ppf(0.975,20) * lmod.bse[1]


#	

thsd = sm.stats.multicomp.pairwise_tukeyhsd(coagulation.coag,
    coagulation.diet)
thsd.summary()


#	

fig = thsd.plot_simultaneous()


# ## False Discovery Rate
#	

import faraway.datasets.jsp
jsp = faraway.datasets.jsp.load()
jsp['mathcent'] = jsp.math - np.mean(jsp.math)


#	

sns.boxplot(x="school", y="mathcent", data=jsp)
plt.xticks(fontsize=8, rotation=90)


#	

lmod = smf.ols("mathcent ~ C(school) - 1", jsp).fit()
lmod.sumary()


#	

sm.stats.anova_lm(smf.ols("mathcent ~ C(school)", jsp).fit())


#	

from statsmodels.stats.multitest import multipletests
reject, padj, _, _ = multipletests(lmod.pvalues, 
    method="bonferroni")
lmod.params[reject]


#	

selsch = np.argsort(lmod.pvalues)[np.sort(lmod.pvalues) < \
    np.arange(1,50)*0.05/49]
lmod.params.index[selsch]


#	

reject, padj, _, _ = multipletests(lmod.pvalues, method="fdr_bh")
lmod.params[reject]


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

    