# # Missing Data
# ## Types of Missing Data
# ## Representation and Detection of Missing Values
#	

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faraway.utils 


#	

import faraway.datasets.chmiss
chmiss = faraway.datasets.chmiss.load()
chmiss.index = chmiss.zip
chmiss.drop(columns=['zip'],inplace=True)
chmiss.head()


#	

chmiss.describe().round(2)


#	

chmiss.isna().sum(axis=0).to_frame().T


#	

chmiss.isna().sum(axis=1).to_frame().T


#	

plt.imshow(~chmiss.isna(), aspect='auto')
plt.xlabel("variables")
plt.xticks(np.arange(0,6), chmiss.columns)
plt.ylabel("cases")


# ## Deletion
#	

import faraway.datasets.chredlin
chredlin = faraway.datasets.chredlin.load()
lmod = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)',
     chredlin).fit()
lmod.sumary()


#	

lmodm = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)', 
    chmiss).fit()
lmodm.sumary()


#	

lmodm.bse/lmod.bse


# ## Single Imputation
#	

cmeans = chmiss.mean(axis=0); cmeans.to_frame().T.round(2)


#	

mchm = chmiss.copy()
mchm.race.fillna(cmeans['race'],inplace=True)
mchm.fire.fillna(cmeans['fire'],inplace=True)
mchm.theft.fillna(cmeans['theft'],inplace=True)
mchm.age.fillna(cmeans['age'],inplace=True)
mchm.income.fillna(cmeans['income'],inplace=True)
imod = smf.ols(
    'involact ~ race + fire + theft + age + np.log(income)',
     mchm).fit()
imod.sumary()


#	

lmodr = smf.ols(
    'race ~ fire + theft + age + np.log(income)', 
    chmiss).fit()
mv = chmiss.race.isna()
lmodr.predict(chmiss)[mv]


#	

def logit(x): return(np.log(x/(1-x)))
def ilogit(x): return(np.exp(x)/(1+np.exp(x)))


#	

lmodr = smf.ols(
   'logit(race/100) ~ fire + theft + age + np.log(income)',
    chmiss).fit()
(ilogit(lmodr.predict(chmiss))*100)[mv]


#	

chredlin.race.iloc[np.where(chmiss.race.isna())]


# ## Multiple Imputation
#	

import statsmodels.imputation.mice as smi
imp = smi.MICEData(chmiss)
fm = 'involact ~ race + fire + theft + age + np.log(income)'
mmod = smi.MICE(fm, sm.OLS, imp)
results = mmod.fit(10, 50)
print(results.summary())


# ## Discussion
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

    