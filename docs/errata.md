# Errata for Linear Model with Python

- p 41: Last code chunk on the page should read:

```
lmoda = smf.ols('Species ~ Area', 
        galapagos).fit()
lmoda.sumary()
```

with output:

```
           coefs stderr tvalues pvalues
Intercept 63.783 17.524    3.64  0.0011
Area       0.082  0.020    4.16  0.0003

n=30 p=2 Residual SD=91.732 R-squared=0.38
```

The p-value of 0.0003 indicates strong statistical significance.
