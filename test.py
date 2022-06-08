import numpy as np
a = np.zeros((1,400,400,3))
a = np.squeeze(a,axis =0)
#convert from HWC to CHW
a = np.transpose(a,(2,0,1))
print(a.shape)