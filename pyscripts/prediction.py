# # Prediction
# ## Confidence Intervals for Predictions
# ## Predicting Body Fat
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils


#	

import faraway.datasets.fat
fat = faraway.datasets.fat.load()


#	

fat.insert(0,'Intercept',1)
x0 = fat.iloc[:,np.r_[0,4:7,9:19]].median()
pd.DataFrame(x0).T


#	

lmod = smf.ols('brozek ~ age + weight + height + neck + \
    chest + abdom + hip + thigh + knee + ankle + biceps + \
    forearm + wrist', fat).fit()
x0 @ lmod.params


#	

lmod.predict(x0)


#	

lmod.get_prediction(x0).summary_frame()


#	

x1 = fat.iloc[:,np.r_[0,4:7,9:19]].quantile(0.95)
pd.DataFrame(x1).T


#	

lmod.get_prediction(x1).summary_frame()


# ## Autoregression
#	

import faraway.datasets.air
air = faraway.datasets.air.load()
plt.plot(air['year'], air['pass'])
X = pd.DataFrame({'Intercept':1, 'year':air['year']})
y = np.log(air['pass'])
lmod = sm.OLS(y,X).fit()
plt.plot(air['year'],np.exp(lmod.predict()))


#	

air['lag1'] = np.log(air['pass']).shift(1)
air['lag12'] = np.log(air['pass']).shift(12)
air['lag13'] = np.log(air['pass']).shift(13)
airlag = air.dropna()


#	

X = airlag.loc[:,('lag1','lag12','lag13')]
X.insert(0,'Intercept',1)
y = np.log(airlag['pass'])
lmod = sm.OLS(y,X).fit()
lmod.sumary()


#	

plt.plot(air['year'], air['pass'])
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.plot(airlag['year'],np.exp(lmod.predict()),linestyle='dashed')


#	

z = np.log(air['pass'].iloc[[-1,-12,-13]]).values
z


#	

x0 = pd.DataFrame([{"const":1,"lag1": z[0], "lag12": z[1], 
    "lag13": z[2]}])
lmod.get_prediction(x0).summary_frame()


# ## What Can Go Wrong with Predictions?
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

    