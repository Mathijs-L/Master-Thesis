# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:41:57 2022

@author: mlenderi
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)

def linear_data(n_weeks):
    series = range(0,n_weeks)
    return pd.DataFrame(series)


df = linear_data(326)

plt.plot(df)
plt.show()
