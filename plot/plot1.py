#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:09:40 2019

@author: haristhemistocleous
"""
# Clean Memory before rerunning 
for name in dir():
    if not name.startswith('_'): del globals()[name]
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

# Preview Options
pd.options.display.max_columns = None
# To plot pretty figures


###### Crossvalidation

# Clean Memory before rerunning
for name in dir():
    if not name.startswith('_'): del globals()[name]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Preview Options
pd.options.display.max_columns = None
# To plot pretty figures

plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = "Set3"




N = 4
Classification1_Means = (64, 64, 67, 61)
Classification2_Means = (81, 69, 69, 63)
Classification1_Std = (3, 17, 19, 5)
Classification2_Std = (7, 27, 42, 10)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, Classification1_Means, width, yerr=Classification1_Std,
              alpha=0.65, label="Classification 1")
p2 = plt.bar(ind+width, Classification2_Means, width,
              yerr=Classification2_Std, alpha=0.65, label="Classification 2")
#plt.grid()
plt.ylabel('Accuracy')
plt.title('Crossvalidation Accuracy')

plt.xticks(ind + width / 2, ('DNN', 'SVM', 'RF', 'DT'))
plt.yticks(np.arange(0, 81, 10))
plt.legend()

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 20,
                '%d' % int(height)+"%",
                ha='center', va='bottom', fontsize=17, color='black')

autolabel(p1)
autolabel(p2)
plt.tight_layout()
plt.savefig("Figure.jpg")
plt.show()
