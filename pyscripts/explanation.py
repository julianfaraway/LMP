# # Explanation
# ## Simple Meaning
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
lmod = smf.ols(
    'Species ~ Area + Elevation + Nearest + Scruz  + Adjacent', 
    galapagos).fit()
lmod.sumary()


#	

lmodr = smf.ols('Species~Elevation',galapagos).fit()
lmodr.sumary()


#	

x0 = galapagos.mean()
xdf = pd.concat([x0,x0],axis=1).T
xrange = [np.min(galapagos.Elevation),np.max(galapagos.Elevation)]
xdf['Elevation'] = xrange


#	

plt.scatter(galapagos.Elevation, galapagos.Species)
plt.xlabel("Elevation")
plt.ylabel("Species")
plt.plot(xrange,lmodr.predict(xdf),"-")
plt.plot(xrange,lmod.predict(xdf),"--")


# ## Causality
# ## Designed Experiments
# ## Observational Data
#	

import faraway.datasets.newhamp
newhamp = faraway.datasets.newhamp.load()
newhamp.groupby('votesys').agg({'Obama': sum, 'Clinton': sum})


#	

newhamp['trt'] = np.where(newhamp.votesys == 'H',1,0)
lmodu = smf.ols('pObama ~ trt',newhamp).fit()
lmodu.sumary()


#	

lmodz = smf.ols('pObama ~ trt+Dean',newhamp).fit()
lmodz.sumary()


#	

lmodc = smf.ols('Dean ~ trt',newhamp).fit()
lmodc.sumary()


# ## Matching
#	

sg = newhamp.Dean[newhamp.trt == 1]
bg = newhamp.Dean[newhamp.trt == 0]
ns = len(sg)
mp = np.full([ns,2],-1)
for i in range(ns):
    dist = abs(sg.iloc[i]-bg)
    if(dist.min() < 0.01):
        imin = dist.idxmin()
        mp[i,:] = [sg.index[i], imin]
        bg = bg.drop(index = imin)
mp = mp[mp[:,0] > -1,:]
mp


#	

newhamp.iloc[[3, 212],[0,6,11]]


#	

sy = newhamp.pObama[newhamp.trt == 1]
by = newhamp.pObama[newhamp.trt == 0]
bg = newhamp.Dean[newhamp.trt == 0]
plt.scatter(bg, by, marker="^", s=2)
plt.scatter(sg, sy, marker="o", s=2)
plt.xlabel("Dean proportion")
plt.ylabel("Obama proportion")
for i in range(len(mp)):
    plt.plot([sg.loc[mp[i,0]], bg.loc[mp[i,1]]], 
             [sy.loc[mp[i,0]], by.loc[mp[i,1]]])


#	

pdiff = sy.loc[mp[:,0]].ravel()-by.loc[mp[:,1]].ravel()
sp.stats.ttest_1samp(pdiff,0)


#	

plt.scatter(sg.loc[mp[:,0]], pdiff)
plt.axhline(0)
plt.xlabel('Proportion voted for Dean')
plt.ylabel('Digital vs. manual difference')


# ## Covariate Adjustment
#	

plt.scatter(bg, by, marker="^", s=2)
plt.scatter(sg, sy, marker="o", s=2)
plt.xlabel("Dean proportion")
plt.ylabel("Obama proportion")
for i in range(len(mp)):
    plt.plot([sg.loc[mp[i,0]], bg.loc[mp[i,1]]], 
             [sy.loc[mp[i,0]], by.loc[mp[i,1]]],color="0.75")
drange = [0.1,0.6]
x0 = pd.DataFrame({"trt": (0,0), "Dean": drange})
x1 = pd.DataFrame({"trt": (1,1), "Dean": drange})
plt.plot(drange,lmodu.predict(x0))
plt.plot(drange,lmodu.predict(x1), linestyle="dashed")
plt.plot(drange,lmodz.predict(x0))
plt.plot(drange,lmodz.predict(x1), linestyle="dashed")


# ## Qualitative Support for Causation
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

    