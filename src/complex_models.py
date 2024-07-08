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
import numpy as np
import networkx as nx
from tqdm import tqdm
from src.utils import mse, mape
from scipy.integrate import odeint

class Threshold_ParamTracker:
    def __init__(self):
        self.best_loss = np.inf
        self.best_threshold = None

    def threshold_loss(self, threshold, graph, initial_infected, model, real_infected, loss = mse):
        max_steps = len(real_infected) - 1  
        S, I = model(graph, initial_infected, threshold, max_steps)
        current_loss = loss(real_infected, I)
        
        # Print loss and parameters
        print(f"Loss (MSE) at iteration with $\\theta = $ {threshold}: {current_loss}")
        
        # Check if the current loss is the best
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_threshold = threshold 
        return current_loss
    
def si_threshold_model(graph, initial_infected, threshold, max_steps):

    # initialitzatiion
    for node in graph.nodes():
        graph.nodes[node]['state'] = 'S'
    for node in initial_infected:
        graph.nodes[node]['state'] = 'I'

    S = [len([node for node in graph.nodes() if graph.nodes[node]['state'] == 'S'])]
    I = [len([node for node in graph.nodes() if graph.nodes[node]['state'] == 'I'])]


    # contagion process: when a susceptible node sees a fraction of infected neighbors that is above a threshold θ
    for step in range(max_steps):
        new_state = {}
        
        for node in graph.nodes():
            if graph.nodes[node]['state'] == 'S':
                total_neighbors = [n for n in graph.neighbors(node)]
                infected_neighbors = [n for n in graph.neighbors(node) if graph.nodes[n]['state'] == 'I']
                if len(total_neighbors) > 0:
                    if len(infected_neighbors)/len(total_neighbors) >= threshold:
                        new_state[node] = 'I'
            # elif graph.nodes[node]['state'] == 'I':
            #     if np.random.rand() < gamma:
            #         new_state[node] = 'R'
        
        for node, state in new_state.items():
            graph.nodes[node]['state'] = state
        
        S.append(len([node for node in graph.nodes() if graph.nodes[node]['state'] == 'S']))
        I.append(len([node for node in graph.nodes() if graph.nodes[node]['state'] == 'I']))
        # R.append(len([node for node in graph.nodes() if graph.nodes[node]['state'] == 'R']))
        
        if I[-1] == 0: # no more infected people
            print("Early stop! No more people is infected!")
            break
    return S, I

