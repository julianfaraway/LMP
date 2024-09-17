# # Inference
# ## Hypothesis Tests to Compare Models
# ## Testing Examples
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils


#	

import faraway.datasets.galapagos
galapagos = faraway.datasets.galapagos.load()
lmod = smf.ols('Species ~ Area + Elevation + Nearest + \
        Scruz + Adjacent', galapagos).fit()


#	

lmod.fvalue, lmod.f_pvalue


#	

lmodr = smf.ols('Species ~ 1', galapagos).fit()
lmod.compare_f_test(lmodr)


#	

lmod.centered_tss, lmod.ssr


#	

lmod.df_model, lmod.df_resid


#	

lmod.mse_model, lmod.mse_resid


#	

lmod.mse_model/ lmod.mse_resid


#	

1-sp.stats.f.cdf(lmod.fvalue, lmod.df_model, lmod.df_resid)


#	

lmods = smf.ols('Species ~ Elevation + Nearest + \
        Scruz + Adjacent', galapagos).fit()
sm.stats.anova_lm(lmods,lmod)


#	

lmod.sumary()


#	

lmoda = smf.ols('Species ~ Area', 
        galapagos).fit()
lmoda.sumary()


#	

lmods = smf.ols('Species ~ Elevation + Nearest + Scruz', 
        galapagos).fit()
sm.stats.anova_lm(lmods,lmod)


#	

lmods = smf.ols('Species ~ I(Area+Adjacent) + \
        Elevation + Nearest + Scruz', galapagos).fit()
sm.stats.anova_lm(lmods,lmod)


#	

lmod = smf.glm('Species ~ Area + Elevation + Nearest + \
        Scruz + Adjacent', galapagos).fit()
lmods = smf.glm('Species ~ Area + Nearest + Scruz + \
        Adjacent', offset=(0.5*galapagos['Elevation']), 
        data=galapagos).fit()
fstat = (lmods.deviance-lmod.deviance)/ \
        (lmod.deviance/lmod.df_resid)
pvalue = 1-sp.stats.f.cdf(fstat, 1, lmod.df_resid)
fstat, pvalue


#	

lmod = smf.ols('Species ~ Area + Elevation + Nearest + \
        Scruz  + Adjacent', galapagos).fit()
tstat=(lmod.params['Elevation']-0.5)/lmod.bse['Elevation']
tstat, 2*sp.stats.t.cdf(tstat, lmod.df_resid)


#	

tstat**2


# ## Permutation Tests
#	

lmod = smf.ols('Species ~ Nearest + Scruz', 
        galapagos).fit()


#	

fstats = np.zeros(4000)
np.random.seed(123)
for i in range(0,4000):
    galapagos['ysamp'] = np.random.permutation(
                np.copy(galapagos['Species']))
    lmodi = smf.ols('ysamp ~ Nearest + \
                Scruz', galapagos).fit()
    fstats[i] = lmodi.fvalue
np.mean(fstats > lmod.fvalue)


#	

lmod.tvalues[2], lmod.pvalues[2]


#	

tstats = np.zeros(4000)
np.random.seed(123)
for i in range(0, 4000):
    galapagos['ssamp'] = np.random.permutation(galapagos.Scruz)
    lmodi = smf.ols('Species ~ Nearest + ssamp', 
                galapagos).fit()
    tstats[i] = lmodi.tvalues[2]
np.mean(np.fabs(tstats) > np.fabs(lmod.tvalues[2]))


# ## Sampling
# ## Confidence Intervals for $\beta$
#	

lmod = smf.ols('Species ~ Area + Elevation + Nearest + \
        Scruz  + Adjacent', galapagos).fit()


#	

qt = np.array(sp.stats.t.interval(0.95,24))
lmod.params[1] + lmod.bse[1]*qt


#	

lmod.params[5] + lmod.bse[5]*qt


#	

lmod.conf_int()


# ## Bootstrap Confidence Intervals
#	

np.random.choice(np.arange(10),10)


#	

np.random.seed(123)
breps = 4000
coefmat = np.empty((breps,6))
resids = lmod.resid
preds = lmod.predict()
for i in range(0,breps):
    galapagos['ysamp'] = preds + np.random.choice(resids,30)
    lmodi = smf.ols('ysamp ~ Area + Elevation + \
        Nearest + Scruz  + Adjacent', galapagos).fit()
    coefmat[i,:] = lmodi.params
coefmat = pd.DataFrame(coefmat, columns=("intercept", 
        "area","elevation","nearest","Scruz","adjacent"))
coefmat.quantile((0.025,0.975))


#	

coefmat.area.plot.density()
xci = coefmat.area.quantile((0.025,0.975)).ravel()
plt.axvline(x=xci[0], linestyle='--')
plt.axvline(x=xci[1], linestyle='--')            


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

    