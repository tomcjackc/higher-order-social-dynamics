#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:31:24 2022

@author: Marius
"""

import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import random
import networkx as nx
import pandas as pd
import random
import collections
import matplotlib.pyplot as plt
import os
import json
from time import time



datasetlist = ['InVS13','InVS15','LH10','LyonSchool','SFHH','Thiers13']

columns_names = ['ID', 'role'] 
for dataset in ['InVS13']:
    df_meta = pd.read_table("metadata/metadata_"+dataset+".dat", names = columns_names, usecols = [0])
    
    df_copresence = pd.read_table('co-presence/tij_pres_'+dataset+'.dat', names = \
                                  ['time', 'pers1', 'pers2'], delimiter = '\ ',\
                                      engine='python')
    
    
    list_times = df_copresence['time'].drop_duplicates()
    
    
    dic = {j: i for i, j in enumerate(df_meta['ID'])}


#def list_nodes(metadata):
    