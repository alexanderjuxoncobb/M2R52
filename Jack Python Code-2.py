#!/usr/bin/env python
# coding: utf-8

# Stock Market Prediction with the Kalman Filter

# In[ ]:


from math import pi
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from datetime import date
from datetime import datetime
from pykalman import KalmanFilter
import datetime
import pickle
import warnings
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np
import pandas as pd
import seaborn as sns
import random


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:



df = pd.read_csv(
    '/Users/macbook/Downloads/FTSE100_Covid19_6MonthData.csv', 
    header=0,
    names=["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"],
    index_col="Date", 
    parse_dates=True
)
    


# In[61]:


KF = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = df['Close'].values[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=0.1)


# In[62]:


state_means,_ = KF.filter(df[['Close']])
state_means = state_means.flatten()


# In[63]:


df["date"] = pd.to_datetime(df.index)

mids = (df.Open + df.Close)/2
spans = abs(df.Close-df.Open)

inc = df.Close > df.Open
dec = df.Open > df.Close
w = 12*60*60*1000 # Half day in ms (for a consistent bar width)

plot = figure(x_axis_type="datetime", plot_width=1000,y_axis_label = "Price", x_axis_label = "Date")

plot.segment(df.date, df.High, df.date, df.Low, color="black")
plot.rect(df.date[inc], mids[inc], w, spans[inc], fill_color='green', line_color="green")
plot.rect(df.date[dec], mids[dec], w, spans[dec], fill_color='red', line_color="red")
plot.line(df.date,state_means, line_width=1,line_color = 'blue', legend_label="Kalman filter")
plot.scatter(df.date,state_means, line_width=1,line_color = 'blue')

p = figure(x_axis_type="datetime", plot_width=1000,y_axis_label = "Price", x_axis_label = "Date")

p.scatter(df.date, df.Close, line_width=1, color = 'red')
p.line(df.date, df.Close, line_width=1, line_color = 'red', legend_label = "Actual Daily Close")
p.line(df.date,state_means, line_width=1,line_color = 'blue', legend_label="Kalman filter")
p.scatter(df.date,state_means, line_width=1,line_color = 'blue')


# Generally, to optimise our Kalman Filter, we would find the maximum likelihood estimate of our parameters (with respect to the data already obtained) and use these parameters in the Kalman Filter implementation. This is not the aim of our project, however for those interested the method can be found at (https://www.adrian.idv.hk/2019-08-18-kalman/) (SHOULD BE GIVEN AS A REFERENCE, NOT LINK HERE) or it can be easily calculated using R-Studio. We have used the defualt values given in the Pykalman package. 
# 
# Implementing the algorithm (REFERENCE TO THE GITHUB REPO) (for each day over the 6 month period): 
# 
# If the Kalman Filter predicts an increase in price then Buy 
# 
# If the Kalman Filter predicts a decrease in price then Sell
# 
# We have obtain a total profit margin of 294.49%, which far out performs naive approaches to trading. Especially given the volatility of the period we have modelled over. 
# 

# In[86]:


show(plot)


# In[87]:


show(p)


# In[81]:


daily_profit = []

for i in range(len(state_means)-1):
    if state_means[i] < state_means[i+1]:
        daily_profit.append(100 * (df.Close[i+1] - df.Close[i])/df.Close[i])
    else:
        daily_profit.append(100 * (df.Close[i] - df.Close[i+1])/df.Close[i])
        
def profit(initial_investment, profit_array):
    for i in profit_array:
        initial_investment += initial_investment * i/100
    return initial_investment

print(f"Gross from £1000 = £{profit(1000, daily_profit):.2f}")
print(f"Profit = £{profit(1000, daily_profit) - 1000:.2f}")
print(f"Profit Margin = {(profit(1000, daily_profit) - 1000)/10:.2f}%")
        


# Particle Filter for Estimating pi

# In[24]:


N = 500 # number of points
radius = 1.
area = (2*radius)**2

pts = uniform(-1, 1, (N, 2))

# distance from (0,0) 
dist = np.linalg.norm(pts, axis=1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)

# plot results
plt.scatter(pts[in_circle,0], pts[in_circle,1], 
            marker=',', edgecolor='k', s=1)
plt.scatter(pts[~in_circle,0], pts[~in_circle,1], 
            marker=',', edgecolor='r', s=1)
plt.axis('equal')
plt.title(f'Particle Filter for N = {N}')

print(f'mean pi(N={N})= {pi:.4f}')
print(f'err  pi(N={N})= {np.pi-pi:.4f}')


# In[25]:


N = 1000 # number of points
radius = 1.
area = (2*radius)**2

pts = uniform(-1, 1, (N, 2))

# distance from (0,0) 
dist = np.linalg.norm(pts, axis=1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)

# plot results
plt.scatter(pts[in_circle,0], pts[in_circle,1], 
            marker=',', edgecolor='k', s=1)
plt.scatter(pts[~in_circle,0], pts[~in_circle,1], 
            marker=',', edgecolor='r', s=1)
plt.axis('equal')
plt.title(f'Particle Filter for N = {N}')

print(f'mean pi(N={N})= {pi:.4f}')
print(f'err  pi(N={N})= {np.pi-pi:.4f}')


# In[27]:


N = 10000 # number of points
radius = 1.
area = (2*radius)**2

pts = uniform(-1, 1, (N, 2))

# distance from (0,0) 
dist = np.linalg.norm(pts, axis=1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)

# plot results
plt.scatter(pts[in_circle,0], pts[in_circle,1], 
            marker=',', edgecolor='k', s=1)
plt.scatter(pts[~in_circle,0], pts[~in_circle,1], 
            marker=',', edgecolor='r', s=1)
plt.axis('equal')
plt.title(f'Particle Filter for N = {N}')

print(f'mean pi(N={N})= {pi:.4f}')
print(f'err  pi(N={N})= {np.pi-pi:.4f}')


# In[85]:


import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform 

N = 100000  # number of points
radius = 1.
area = (2*radius)**2

pts = uniform(-1, 1, (N, 2))

# distance from (0,0) 
dist = np.linalg.norm(pts, axis=1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)  # Since (area of circle)/(area of square) = pi*4

# plot results
plt.scatter(pts[in_circle,0], pts[in_circle,1], 
            marker=',', edgecolor='k', s=1)
plt.scatter(pts[~in_circle,0], pts[~in_circle,1], 
            marker=',', edgecolor='r', s=1)
plt.axis('equal')
plt.title('Particle Filter for N = {N}')

print(f'mean pi(N={N})= {pi:.4f}')
print(f'err  pi(N={N})= {np.pi-pi:.4f}')

# Code from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

