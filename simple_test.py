#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:32:49 2023

@author: Marius
"""
from enum import unique
import importlib
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools
from HONG import *
import HONG
import csv
from tqdm import tqdm
import csv
from numpy import genfromtxt
import matplotlib as mpl
import seaborn as sb
import pandas as pd


# test code

output_fname = 'test'
edges = [[1,2,3], [1,2], [2,3], [1,4]]
committed_nodes = [3, 4]
uncommitted_nodes = [1,2]

H = HigherOrderNamingGame(rule='Unanimous')
H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=1, q=1)
H.add_naming_game_node(committed_nodes, ['B'], True, beta=1, q=1)

H.add_edges_from(edges)

H.move_speaker(3, 0)

#run_ensemble_experiment(0.03, 0.27, 0.27, 1, 10**2, 'LyonSchool', q_non_committed=1, q_committed=1)

