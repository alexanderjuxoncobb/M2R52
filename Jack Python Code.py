#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[82]:


show(plot)


# In[65]:


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
        


# In[ ]:





# Particle filter for estimating pi

# BRIEF DESCRIPTION OF A PARTICLE FILTER
# 
# A particle filter is similar to the aforementioned Kalman Filter, and is a technique for estimating the state of a dynamic system recursively. We take the current belief for the state of a system and update it recursively based on so-called motion information of control commands and our observations of the true state of the system. It differs from the Kalman Filter by relaxing assumptions about the state-space model and the state distributions. This means the Particle Filter does not perform well when applied to high-dimensional systems, but has the advantage that it can deal with non-linear systems, drops the assumption of Gaussian noise (we can use any arbitrary probability distributions) and it uses a non-parametric form, making it far easier to explain and understand. 
# 
# Below we have a simple example of a particle filter being used to estimate the value of pi. We know that the area of the circle is pi*r^2 and the area of the square is width^2 = (2*r)^2. Hence, we see that pi = 4 * (area of circle)/ (area of square). In our model this is the ratio of particles inside the circle to outside the circle. We can see that as the number of particles increases, so does the accuracy of our measurement and it converges uniformly to the true value of pi. 
# 
# 
# 
# 
# 
# Jack, can you put the maths here into latex please. I dont have it on my computer - sorry. 

# GENERAL PARTICLE FILTER ALGORITHM
# 
# 1. Randomly Generate a Set Of Particles 
# 
# Each particle will have a weight (which is it's probability) indicating how likely it is that it matches the true state of the system. We initialise all particles with the same weight. 
# 
# 2. Predict The Next State Of The Particles 
# 
# Move the particles based on how we predict the real system will behave.
# 
# 3. Update The System
# 
# Update the weighting of each particle based on a measurement of the real system. The closer a particle to the measurement the higher weight it is given. 
# 
# 4. Resample 
# 
# Discard highly improbable particles with more probable ones, at a confidence level decided by the experimentor (a 95% confidence interval is standard). 
# 
# 5. Repeat steps 2-4
# 
#                                   

# In[ ]:





# I think the best way to present the pi estimation is by showing the code below (once) and then showing all 4 images together, along with their respective errors. 

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

