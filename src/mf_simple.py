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

def sir_model_ODE(y, t, N, params):
    beta, gamma = params
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def sirs_model_ODE(y, t, N, params):
    beta, gamma, delta = params
    S, I, R = y
    dSdt = -beta * S * I / N + delta * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - delta * R
    return dSdt, dIdt, dRdt

