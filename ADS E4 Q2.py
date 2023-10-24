# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:39:27 2023

@author: Dell
"""

# Import Numpy, Matplotlib and Pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crate DataFrame for CSV files

BCS = pd.read_csv('BCS_ann.csv')
print(BCS)
BP = pd.read_csv('BP_ann.csv')
TSCO = pd.read_csv('TSCO_ann.csv')
VOD = pd.read_csv('VOD_ann.csv')

print(BCS, BP, TSCO, VOD)

plt.figure()

plt.subplot(2, 2, 1)
plt.hist(BCS['ann_return'], label='BARCLAYS', color='green')
plt.ylabel('BARCLAYS')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(BP['ann_return'], label='BP', color='orange')
plt.ylabel('BP')
plt.legend()


plt.subplot(2, 2, 3)
plt.hist(TSCO['ann_return'], label='TESCO', color='red')
plt.ylabel('TESCO')
plt.legend()


plt.subplot(2, 2, 4)
plt.hist(VOD['ann_return'], label='VODAFONE')
plt.ylabel('VODAFONE')
plt.legend()



plt.show()

