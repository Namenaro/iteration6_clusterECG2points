import numpy as np
"""a= np.array([0,5,0,0,4,2,1])
kernel = np.array([0.1,1.0,0.1]) # Here you would insert your actual kernel of any size
a = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, a)
print (a)"""


from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
np.random.seed(280490)
x = np.random.randn(101).cumsum()
y3 = gaussian_filter1d(x, 3)
y6 = gaussian_filter1d(x, 6)
plt.plot(x, 'k', label='original data')
plt.plot(y3, '--', label='filtered, sigma=3')
plt.plot(y6, ':', label='filtered, sigma=6')
plt.legend()
plt.grid()
plt.show()


################################
a = np.array([1,3,9,3,7,1])
max_index = np.argmax(a)
print (max_index)
################################

from scipy.interpolate import interp1d
from scipy import signal

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

y = np.array([1,2,3,4,5,6,10])
#downsampled_y = signal.resample(y, 4)
downsampled_y = downsample(y, 4)
print(downsampled_y)

################################
import torch
from torch import nn
m = nn.AvgPool1d(3, stride=2)
y = np.array([1,2,3,4,5,6,10])

y= m(torch.tensor([[y]])).numpy()
print (np.squeeze(y))
####################################
