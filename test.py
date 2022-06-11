
#%%
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import torch
# Load the data
target_s = pickle.load(open("/root/ClipNeRF_base/logs/fern_test/target_s.pkl", "rb"))
target_s = target_s.detach().cpu().numpy()
# %%
target_s = np.reshape(target_s, (40,40, -1))
plt.imshow(target_s[:,:,:])

# %%
