#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:05:59 2023

@author: Marius
"""

from integrate import *





betas = [0.4]
ps = [0.1]
qs = [0]
social_structures = ['InVS15']
run_length = 10**4
import warnings
warnings.filterwarnings("ignore")

run_multiprocessing_ensemble(ps, betas, run_length, social_structures, qs)
create_csvs_from_outputs(ps, betas, run_length, social_structures, qs)