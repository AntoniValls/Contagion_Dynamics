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
from collections import Counter
import numpy as np

def filter_most_frequent_values(d):
    # Step 1: Count the frequencies of each value in the dictionary
    value_counts = Counter(d.values())
    
    # Step 2: Find the most frequent value(s)
    most_frequent_value = value_counts.most_common(1)[0][0]
    
    # Step 3: Create a new dictionary with only the most frequent value(s)
    filtered_dict = {k: v for k, v in d.items() if v == most_frequent_value}
    
    return filtered_dict

def mape(real, simulated):
    return np.mean(np.abs((simulated - real) / real)) * 100

def mse(real, simulated):
    return np.mean((simulated - real) ** 2)
