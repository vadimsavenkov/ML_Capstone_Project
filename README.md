
# Product demand predictive analysis using time series forecasting methods. 

# Problem Statement

The project goal is to predict product order demand usingtime series forecasting methods.

# Dataset Description

The dataset consists of historical product demand for a manufacturer.

# Source of data 

The data have been downloaded from Kaggle. (https://www.kaggle.com/felixzhao/productdemandforecasting/data)

# Requirements

• Data Analysis and Preparation
• Perform time series analysis for product category using various forecasting techniques
• Run models and compare evaluation metrics

# Project Outcome

This document covers the most popular statistical methods for time series forecasting. It includes data analysis and visualization, statistical functions and interactive plots implemented in python.

# Package instalation check:

Python Version---- 3.7.3 

Numpy Version---- 1.16.3

Scipy Version---- 1.2.1

Scikit-Learn Version---- 0.20.3

Pandas Version---- 0.24.2

Statsmodels Version---- 0.9.0

Pip Version---- 19.0.3

Pmdarima Version---- 1.2.1

# libraries:

import pandas as pd
import numpy as np
import itertools

# Statistical packagies

import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn import metrics

from pmdarima.arima import auto_arima

import pmdarima as pm

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# Plotting libraries

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

# Configure Plotly to be rendered inline in the notebook.

import plotly as py

py.offline.init_notebook_mode(connected=True)

plt.style.use('fivethirtyeight')

# Dependencies

import plotly.graph_objs as go

import ipywidgets as widgets

from scipy import special













