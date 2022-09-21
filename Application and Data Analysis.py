#!/usr/bin/env python
# coding: utf-8

# Perez Bikes Prediction Application 

# In[1]:


import pandas as pd
import numpy as np
import os
import psycopg2 as ps
import pandas.io.sql as sqlio
import matplotlib.pyplot as plt
from datetime import datetime
import sqlalchemy as sqla


# In[2]:


connection = ps.connect(dbname="Perez Bikes",
                       user = "postgres",
                       password = "superman94",
                       host = "localhost",
                       port = "5432")


# In[3]:


pred = pd.read_sql_query('''SELECT * FROM PREDICTION;''', connection)


# In[4]:


# returns the quantity sold per day of all bikes 
b_per_day = pd.read_sql_query('''SELECT order_date, COUNT(quantity) FROM PREDICTION GROUP BY order_date ORDER BY order_date;''', connection)
b_per_day["order_date"] = pd.to_datetime(b_per_day["order_date"])
b_per_day['year_week'] = b_per_day['order_date'].dt.isocalendar().week
b_per_day['year'] = b_per_day['order_date'].dt.isocalendar().year
b_per_day


# In[5]:


# add a column to the pred table that contains the week of the year and year of the order 
pred["order_date"] = pd.to_datetime(pred["order_date"])
pred['year_week'] = pred['order_date'].dt.isocalendar().week
pred['year'] = pred['order_date'].dt.isocalendar().year
pred


# In[6]:


import ipywidgets as w    


# In[7]:


# create a tab menu for ts graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output
 
# output variables
op11 = w.Output()
op12 = w.Output()
op13 = w.Output()
op14 = w.Output()
op15 = w.Output()


# tab setup 
tab = w.Tab(children = [op11, op12, op13, op14, op15])
tab.set_title(0,'2011')
tab.set_title(1, '2012')
tab.set_title(2, '2013')
tab.set_title(3, '2014')
tab.set_title(4, '2015')
display(tab)


# output commands 
with op11:
    clear_output(wait=True)
    fig1, ax1 = plt.subplots()
    ts_2011 = b_per_day.loc[(b_per_day['year'] == 2011)]
    ts2011_plot = ax1.plot(ts_2011['year_week'], ts_2011['count'])
    plt.title('2011')
    ax1.set_ylabel('Quantity')
    ax1.set_xlabel('Week')
    plt.show(fig1)
    
with op12:
    fig1, ax2 = plt.subplots()
    ts_2012 = b_per_day.loc[(b_per_day['year'] == 2012)]
    ts_2012_plot = ax2.plot(ts_2012['year_week'], ts_2012['count'])
    plt.title('2012')
    ax2.set_ylabel('Quantity')
    ax2.set_xlabel('Week')
    plt.show(fig1)

    
with op13:
    fig1, ax3 = plt.subplots()
    ts_2013 = b_per_day.loc[(b_per_day['year'] == 2013)]
    ts_2013_plot = ax3.plot(ts_2013['year_week'], ts_2013['count'])
    plt.title('2013')
    ax3.set_ylabel('Quantity')
    ax3.set_xlabel('Week')
    plt.show(fig1)
    
with op14:
    fig4, ax4 = plt.subplots()
    ts_2014 = b_per_day.loc[(b_per_day['year'] == 2014)]
    ts_2014_plot = ax4.plot(ts_2014['year_week'], ts_2014['count'])
    plt.title('2014')
    ax4.set_ylabel('Quantity')
    ax4.set_xlabel('Week')
    plt.show(fig4)

with op15:
    fig5, ax5 = plt.subplots()
    ts_2015 = b_per_day.loc[(b_per_day['year'] == 2015)]
    ts_2015_plot = ax5.plot(ts_2015['year_week'], ts_2015['count'])
    plt.title('2015')
    ax5.set_ylabel('Quantity')
    ax5.set_xlabel('Week')
    plt.show(fig5)
    
# close all figures
# plt.close()


# In[8]:


# XY scatter plot (x: customer id, y: number of bikes per transaction)
xy_table = pd.read_sql_query('''SELECT customer_id, COUNT(quantity) FROM orders_updated GROUP BY customer_id;''', connection)


# In[9]:


xy_table.plot.scatter(x='customer_id', y='count')


# In[10]:


# bar graph (y: the amount of that bike sold that week, x: model name of bike)
# [bike.id should be transformed here to make it easier to read]
bar_graph = pd.read_sql_query('''SELECT model, order_date, quantity FROM prediction''', connection)
bar_graph['order_date'] = pd.to_datetime(bar_graph["order_date"])
bar_graph['year_week'] = bar_graph['order_date'].dt.isocalendar().week
bar_graph['year'] = bar_graph['order_date'].dt.isocalendar().year
bar_graph.groupby(['year','model']).count()
bg_update=bar_graph.drop(columns=['order_date', 'year_week'], axis = 1)
bg_pivot = bg_update.pivot_table(index='model', columns= 'year', values='quantity', aggfunc='sum')
bg_pivot.plot.barh(figsize=(16,20))
plt.yticks(fontsize=10, rotation=30)
plt.legend(bbox_to_anchor=(0.5,1))


# In[11]:


# linear regression algorithm 
# args should be two dates: following monday and next saturday
# function should return an array: [bike 1's name : quantity; b2 : q; b3 : q]
import xgboost as xgb


# In[12]:


import sklearn


# In[13]:


# cleaning data to prep for algorithm
xgb_table = pred.drop(['order_id', 'order_date'], axis=1)


# In[14]:


xgb_table = xgb_table[['product_id', 'model', 'quantity', 'year_week', 'year']]
xgb_table


# In[15]:


# sorting the above table 
xgb_groupby = xgb_table.groupby(['year', 'year_week', 'product_id']).sum().reset_index()
xgb_groupby


# In[16]:


X = xgb_groupby['product_id']
y = xgb_groupby['quantity']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


import numpy as np


# In[21]:


# convert X_train to a 2D array 
X_train.to_numpy()
X_train


# In[22]:


X_train = np.array(X_train).reshape(-1,1)


# In[23]:


# converting X_test to a 2D array 
X_test.to_numpy()
X_test = np.array(X_test).reshape(-1,1)


# In[24]:


# linear regression method
lr = LinearRegression()
lr.fit(X_train, y_train)
Y_pred= lr.predict(X_test)


# In[25]:


from sklearn import metrics


# In[26]:


print("MAE: ", metrics.mean_absolute_error(y_test, Y_pred))
print("MSE: ", metrics.mean_squared_error(y_test, Y_pred))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))


# In[27]:


# XGBoost method 
xgb_regressor = xgb.XGBRegressor(booster = 'gblinear')
xgb_regressor.fit(X_train, y_train)


# In[28]:


xgb_pred = xgb_regressor.predict(X_test)


# In[29]:


print("RSME: ", np.sqrt(metrics.mean_squared_error(y_test, xgb_pred)))


# In[30]:


xgb_pred

