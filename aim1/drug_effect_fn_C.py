#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:38:48 2020

@author: harshparikh
"""
import numpy as np
import pandas as pd
import sklearn.tree as tree

data = pd.read_csv('/Users/harshparikh/Documents/GitHub/iic_causal_inference/aim1/simulator/results/params_cauchy_expit_ARMA16_iter1000.csv',index_col=0)

cols_y = ['b[2]']

def run_reg(cols_y,data):
    for col_y in cols_y:
        data_prime = data.dropna(subset=[col_y])
        
        X = data_prime[['Gender', 'Age', 'marrital',
               'Hx CVA (including TIA)', 'Hx HTN', 'Hx Sz /epilepsy',
               'Hx brain surgery', 'Hx CKD', 'Hx CAD/MI', 'Hx CHF', 'Hx DM',
               'Hx of HLD', 'Hx tobacco (including ex-smokers)',
               'Hx ETOH abuse any time in their life (just when in the hx is mentioned)',
               'Hx other substance abuse, any time in their life',
               'Hx cancer (other than CNS cancer)', 'Hx CNS cancer', 'Hx COPD/ Asthma',
               'premorbid MRS before admission  (modified ranking scale),before admission',
               'SZ at presentation,(exclude non-convulsive seizures) just if it is mentioned in MGH notes (the date is necessary, however,the date is the day of admission at MGH)',
               'hydrocephalus  (either on admission or during hospital course)   QPID',
               'iMV  (initial (on admission) mechanical ventilation)', 'systolic BP',
               'diastolic BP', 'Midline shift with any reason ( Document Date)',
               'Primary systemic dx Sepsis/Shock', 'iGCS-Total', 'iGCS = T?', 'iGCS-E',
               'iGCS-V', 'iGCS-M', 'Worst GCS in 1st 24',
               'Worst GCS Intubation status', 'iGCS actual scores',
               'neuro_dx_Seizures/status epilepticus',
               'prim_dx_Respiratory disorders']]
        
        Y = data_prime[col_y]
    
        