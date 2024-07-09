# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import networkx as nx
from tqdm import tqdm
from src.utils import mse, mape
from scipy.integrate import odeint
    
# MACRO models (ODE based)
def loss_ODE(params, graph, initial_infected, model, real_infected, loss = mse):
    
    # A grid of time points (in days)
    t = np.arange(0., 86.) 

    # Initial state
    N = len(set(graph.nodes()))
    I0, R0 = len(initial_infected), 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    
    ret = odeint(model, y0, t, args=(N, params))
    S, I, R = ret.T
    return loss(real_infected, I)


def ame_sir_model(y, t, k_values, beta, gamma, degree_dist, mean_degree):
    # Unpack the state variables
    S_km, I_km, R_km = y.reshape((3, len(k_values), -1)) # s_km, i_km, r_km matrices
    
    # Number of neighbors in each state
    S_k = np.sum(S_km, axis=1)
    I_k = np.sum(I_km, axis=1)
    R_k = np.sum(R_km, axis=1)
    
    # Compute the total density of each state
    S = np.sum(degree_dist * S_k)
    I = np.sum(degree_dist * I_k)
    R = np.sum(degree_dist * R_k)
    
    # Effective infection probability (mean-field approximation)
    theta = np.sum(degree_dist * I_k) / mean_degree
    
    # Differential equations
    dS_km = np.zeros_like(S_km)
    dI_km = np.zeros_like(I_km)
    dR_km = np.zeros_like(R_km)
    
    for k_idx, k in enumerate(k_values):
        for m in range(k + 1):
            # Rate of change for susceptible nodes
            if m > 0:
                dS_km[k_idx, m] = -beta * m / k * S_km[k_idx, m] * theta + beta * (k - m + 1) / k * S_km[k_idx, m - 1] * theta
            else:
                dS_km[k_idx, m] = -beta * m / k * S_km[k_idx, m] * theta
            
            # Rate of change for infected nodes
            if m < k:
                dI_km[k_idx, m] = beta * m / k * S_km[k_idx, m] * theta - gamma * I_km[k_idx, m]
            else:
                dI_km[k_idx, m] = -gamma * I_km[k_idx, m]
            
            # Rate of change for recovered nodes
            if m < k:
                dR_km[k_idx, m] = gamma * I_km[k_idx, m] + beta * (k - m) / k * S_km[k_idx, m + 1] * theta
            else:
                dR_km[k_idx, m] = gamma * I_km[k_idx, m]

    # Flatten the differential equations for integration
    dydt = np.concatenate([dS_km.flatten(), dI_km.flatten(), dR_km.flatten()])
    return dydt

