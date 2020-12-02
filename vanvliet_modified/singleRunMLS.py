#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:56:43 2018

Run MLS model and plot results

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
"""
# %% single run of model
import MLS_static_fast as mlssf
import matplotlib.pyplot as plt

# SET model settings
model_par = {
    # %% Host Paramaters
    # Host carying capacity
    "K_H": 500.,
    # Host birth rate = 1/TAU_H * (1 + B_H * Helper density)
    "B_H": 0.,
    # Host death rate = (1/TAU_H) * (#Host/K_H) * (1 - D_H * Helper density)
    "D_H": 0.,
    # Number of microbe generations per host generation
    "TAU_H": 1000,
    # %% Microbiome Parameters
    # Helper birth rate = (1-cost)
    "cost": 0.01,
    # %% Vertical Transmission Paramaters
    # Sampling variation (STD of normal distribution)
    "sigmaBirth": 0.02,
    # Density of vertically transmitted inoculum (k=1)
    "n0": 1E-4,
    # %% Horizontal Transmission Paramaters
    # Migration rate between hosts
    "mig": 1E-6,
    # %% initial conditions
    #Fraction helper cells at t=0
    "F0": 0.5,
    #fraction of hosts with helper gene
    "FH0": 0.5,
    #Density of microbes at t=0 (<=1)
    "N0init": 1.,
    #Number of hosts t=0 (use -1 to set to K_H)
    "NUMGROUP": -1,
    # %%time settings
    #Max run time of simulation
    "maxT": 150000,
    #Time step at which microbiome and hosts rates are updated
    "dT": 0.05,
    #Time step at which model state is stored
    "sampleT": 1,
    #Length of time window over which to average helper fraction
    "mav_window": 1000,
    #Length of time window over which to calculate root-mean-square error of helper fraction
    "rms_window": 10000,
    #When RMS error < treshold steady state has been reached and simulation ends 
    "rms_err_treshold": 0.05,
    # %%fixed model parameters
    #Type of sampling
    "sampling": "fixedvar", #truncated normal distribution
    #transition rate between helper cells and neutral cells
    "mu": 1E-9,
    #carrying capacity of microbiome, by definition = 1 (rescaling of units such that beta=delta=k=1)
    "K": 1,
    #Number of bins in histrogram of helper fraction per host
    "numTypeBins": 100
}


Output, InvestmentPerHost = mlssf.single_run_with_plot(model_par)

print(Output.dtype.names)
#plt.plot(Output['F_mav'])
plt.show()