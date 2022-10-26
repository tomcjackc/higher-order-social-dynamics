#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:28:17 2022

@author: Marius
"""

"""
This will be the file that prouces the stationary averages presented in govani's
paper
"""

from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from imports import *
import csv
from tqdm import tqdm
from numpy import genfromtxt

'''
### USE THIS TO MAKE PRETTY PICTURES, DOES NOT WORK FOR MARIUS ###


plt.style.use(mpl_style.mplstyle)
'''


