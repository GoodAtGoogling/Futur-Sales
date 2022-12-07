# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:02:55 2022

@author: fondr
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os

        
train = pd.read_csv('./Data/sales_train.csv',sep=";")
test = pd.read_csv('./Data/test.csv',sep =";")
shops = pd.read_csv('./Data/shops.csv')
items = pd.read_csv('./Data/items.csv',sep = ",")
item_categories = pd.read_csv('./Data/item_categories.csv')


print("Train : " ,"\n",train.dtypes,"\n")
print("Test : " ,"\n",test.dtypes,"\n")
print("Shops : " ,"\n",shops.dtypes,"\n")
print("Items : " ,"\n",items.dtypes,"\n")
print("Item_Cat : " ,"\n",item_categories.dtypes,"\n")


train.isnull().sum()