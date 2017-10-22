# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:26:54 2017

@author: dream_rab04is
"""

import pandas as pd
import numpy as np

test = pd.read_csv("../input/test.csv", low_memory=False)

a = pd.read_csv("XG_0.csv", low_memory=False).Sales
b = pd.read_csv("XG_1.csv", low_memory=False).Sales
c = pd.read_csv("XG_2.csv", low_memory=False).Sales
d = pd.read_csv("XG_3.csv", low_memory=False).Sales
e = pd.read_csv("XG_4.csv", low_memory=False).Sales
f = pd.read_csv("XG_5.csv", low_memory=False).Sales
g = pd.read_csv("XG_6.csv", low_memory=False).Sales

result=[]

for iter in range(0,len(a)):
    r_temp = np.mean([  
                      [a[iter]], 
                      [b[iter]],
                      [c[iter]], 
                      [d[iter]],
                      [e[iter]], 
                      [f[iter]],
                      [g[iter]]                      
                      ])
    
    result.append(r_temp) 

result = pd.DataFrame({"Id": test["Id"], 'Sales': result})
result.to_csv("XG_Aver.csv", index=False)

