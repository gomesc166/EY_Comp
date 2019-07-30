
# coding: utf-8

# # EY Data Science Challenge 2019

# # Feature Engineering - Extrapolation of Destination
# 
# The idea behind this script is to extrapolate the end (x,y) coordinates for each hash group using the sequence of trajectories provided.

# In[1]:


import pandas as pd
import numpy as np
import time


# ## Read in the data

# In[2]:


dtype = {
    "vmax" : np.float64,
    "vmin" : np.float64,
    "vmean" : np.float64,
    "x_entry" : np.float64,
    "y_entry" : np.float64,
    "x_exit" : np.float64,
    "y_exit" : np.float64
}


# In[3]:


# read in the training data
train = pd.read_csv("data_train_mod.csv", dtype=dtype, index_col=0)
#train['time_entry'] = pd.to_datetime(train['time_entry'])
#train['time_exit'] = pd.to_datetime(train['time_exit'])


# In[4]:

"""
def compute_seconds_from_midnight(timestamp):
    return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second


# In[5]:


train["time_entry"] = train["time_entry"].apply(compute_seconds_from_midnight)
train["time_exit"] = train["time_exit"].apply(compute_seconds_from_midnight)
"""

# ## Preprocessing - flattening the data

# In[6]:


N_FEATURES = 10
N_REQUIRED_TRAJECTORIES = 6 # inclusive of the final trajectory

def flatten_hash_group(group, is_train=True):
    
    # construct a new series to store the flattened hash instance
    to_flatten = pd.Series(group["hash"].unique())
    
    # pad the flattened array with zero-values
    n_trajectories_to_pad = N_REQUIRED_TRAJECTORIES - len(group)
    if n_trajectories_to_pad > 0:
        padding = pd.Series([0] * n_trajectories_to_pad * N_FEATURES)
        to_flatten = to_flatten.append(padding, ignore_index=True)
    
    # iterate through the last N_REQUIRED_TRAJECTORIES
    for i in range(max(0, len(group) - N_REQUIRED_TRAJECTORIES), len(group)):
        to_append = pd.Series(group.iloc[i][["x_entry", "y_entry", "time_entry", "x_exit", "y_exit", "time_exit","vel1","time_diff1","vel2","time_diff2"]].values)
        to_flatten = to_flatten.append(to_append, ignore_index=True)

    return to_flatten


# In[56]:


def process_data(df, outfile="train_flat.csv", n_iters_to_write=100, is_train=True):
    
    # overwrite the existing data file
    with open(outfile, "w") as f:
        pass
    
    # chunk through the hash groups, write to outfile
    counter = 0
    data = pd.DataFrame()
    for _, hash_group in df.groupby("hash"):
        feature_vector = flatten_hash_group(hash_group, is_train)
        data = data.append(feature_vector, ignore_index=True)
        
        counter += 1
        if counter % n_iters_to_write == 0:
            print("Counter: " + str(counter) + "... writing to outfile.")
            with open(outfile, "a") as f:
                data.to_csv(f, header=False, index=False)
            data = pd.DataFrame()
    
    # don't forget, the last few groups might not have been written out
    if not df.empty:
        with open(outfile, "a") as f:
            data.to_csv(f, header=False, index=False)
            data = pd.DataFrame()


# In[57]:


tick = time.clock()
train_fe = process_data(train, n_iters_to_write=100)
tock = time.clock()
print(tock - tick)


# ## Check that we've written out the data correctly

# In[60]:

"""
check = pd.read_csv("train_flat.csv", header=None, index_col=None)
check.head(10)
"""

