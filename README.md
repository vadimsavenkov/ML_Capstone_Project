
# Product demand predictive analysis using time series forecasting methods. 

## Problem Statement

The project goal is to predict product order demand usingtime series forecasting methods.

## Dataset Description

The dataset consists of historical product demand for a manufacturer.

## Source of data 

The data have been downloaded from Kaggle. (https://www.kaggle.com/felixzhao/productdemandforecasting/data)

## Requirements

• Data Analysis and Preparation
• Perform time series analysis for product category using various forecasting techniques
• Run models and compare evaluation metrics

## Project Outcome

This document covers the most popular statistical methods for time series forecasting. It includes data analysis and visualization, statistical functions and interactive plots implemented in python.

### Package instalation check:
```python
Python Version---- 3.7.3     
Numpy Version---- 1.16.3    
Scipy Version---- 1.2.1    
Scikit-Learn Version---- 0.20.3    
Pandas Version---- 0.24.2    
Statsmodels Version---- 0.9.0    
Pip Version---- 19.0.3    
Pmdarima Version---- 1.2.1
```

### Libraries
```python
import pandas as pd    
import numpy as np    
import itertools
```

### Statistical packagies
```python
import statsmodels.api as sm  
from statsmodels.tsa.seasonal import seasonal_decompose  
from sklearn import metrics  
from pmdarima.arima import auto_arima  
import pmdarima as pm  
from IPython.core.interactiveshell import InteractiveShell  
InteractiveShell.ast_node_interactivity = "all"
```

### Plotting libraries
```python
import matplotlib  
import matplotlib.pyplot as plt  
import seaborn as sns  
import scipy  
%matplotlib inline  
from matplotlib.pylab import rcParams  
matplotlib.rcParams['axes.labelsize'] = 14  
matplotlib.rcParams['xtick.labelsize'] = 12  
matplotlib.rcParams['ytick.labelsize'] = 12  
matplotlib.rcParams['text.color'] = 'k'
```

### Configure Plotly to be rendered inline in the notebook.
```python
import plotly as py  
py.offline.init_notebook_mode(connected=True)  
plt.style.use('fivethirtyeight')
```

### Dependencies
```python
import plotly.graph_objs as go  
import ipywidgets as widgets  
from scipy import special
```

## Introduction

Time series forecasting is one of the most applied data science techniques in finance, supply chain management and inventory planning and very well established in statistics. The methods are used in this document as follows:

• Linear Smoothing  
• Exponential Smoothing  
• Holt's Linear Trend  
• Holt-Winters Method  
• Seasonal ARIMA(Autoregressive Integrated Moving Average)  

## Data Preprocessing

• Read dataset and explore the features

```python
df = pd.read_csv('HPD2.csv', parse_dates=['Date'])  
df.head()

Product_Code	Warehouse	Product_Category	Date	Order_Demand
0	Product_1507	Whse_C	Category_019	2011-09-02	1250
1	Product_0608	Whse_C	Category_001	2011-09-27	5
2	Product_1933	Whse_C	Category_001	2011-09-27	23
3	Product_0875	Whse_C	Category_023	2011-09-30	5450
4	Product_0642	Whse_C	Category_019	2011-10-31	3
```
Dataset has 5 features:
1. Product Code (categorical): 2160 products  
2. Warehouse (categorical): 4 warehouses  
3. Product category (categorical): 33 categories    
4. Date (date): demand fulfillment date  
5. Order demanded (integer): target demand value  

• Group data by product category and make it in order by count to get the order demand for top three categories
```python
Cat_fltr = df.groupby(['Product_Category']).size().reset_index(name='Cat_count').sort_values(['Cat_count'],ascending=False)
Cat_fltr.head(10)
Product_Category	Cat_count
18	Category_019	445251
4	Category_005	100711
0	Category_001	96841
6	Category_007	81159
20	Category_021	50938
5	Category_006	35098
27	Category_028	28923
10	Category_011	22973
14	Category_015	22437
23	Category_024	20371
```
Top three categories by order demand are Categories 1,5 and 19. The Category_001 is picked in this case for time series analysis.


